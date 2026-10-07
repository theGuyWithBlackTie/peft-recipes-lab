import os
import json
import re
import time
import logging
from typing import Dict, Any, List, Optional, Union, Callable

from evaluation.judge_prompts import (
    SYSTEM_PROMPT_SINGLE_EVAL,
    USER_PROMPT_SINGLE_EVAL,
    SYSTEM_PROMPT_PAIRWISE_EVAL,
    USER_PROMPT_PAIRWISE_EVAL
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMJudge:
    """
    Automated LLM-as-a-Judge evaluator for assessing conversational AI and Hinglish code-mixing.
    Supports OpenAI, Google Gemini, Anthropic, or custom local inference callables.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.1-pro-preview",
        custom_caller: Optional[Callable[[str, str], str]] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "global",
        max_retries: int = 5
    ):
        """
        Initializes the LLM Judge.

        Args:
            model_name (str): Judge model identifier (e.g. 'gemini-2.5-flash', 'gpt-4o-mini', 'claude-3-5-sonnet-20241022').
            custom_caller (Callable, optional): Custom function taking (system_prompt, user_prompt) -> response_str.
            api_key (str, optional): API key for the selected provider.
            project_id (str, optional): GCP project ID for Vertex AI (supports ADC).
            location (str): GCP region for Vertex AI (defaults to 'us-central1').
            max_retries (int): Number of retries on API failure/rate-limit.
        """
        self.model_name = model_name
        self.custom_caller = custom_caller
        self.api_key = api_key
        self.project_id = project_id
        self.location = location
        self.max_retries = max_retries

    def _call_judge_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Routes system prompt and user input to the configured LLM backend with retry logic.
        """
        if self.custom_caller is not None:
            return self.custom_caller(system_prompt, user_prompt)

        for attempt in range(self.max_retries):
            try:
                # 1. OpenAI / Compatible endpoint
                if "gpt" in self.model_name.lower() or "o1" in self.model_name.lower():
                    from openai import OpenAI
                    client = OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY"))
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                        response_format={"type": "json_object"}
                    )
                    return response.choices[0].message.content

                # 2. Google Gemini / Vertex AI
                elif "gemini" in self.model_name.lower():
                    try:
                        from google import genai
                        from google.genai import types

                        project = self.project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
                        if not project:
                            try:
                                import subprocess, shutil
                                gcloud_bin = shutil.which("gcloud") or os.path.expanduser("~/google-cloud-sdk/bin/gcloud")
                                if os.path.exists(gcloud_bin):
                                    p = subprocess.run([gcloud_bin, "config", "get-value", "project"], capture_output=True, text=True, timeout=2)
                                    if p.returncode == 0 and p.stdout.strip():
                                        project = p.stdout.strip()
                            except Exception:
                                pass

                        use_vertex = bool(project or self.project_id or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "true" or not (self.api_key or os.environ.get("GEMINI_API_KEY")))

                        if use_vertex:
                            client = genai.Client(
                                vertexai=True,
                                project=project,
                                location=self.location
                            )
                        else:
                            client = genai.Client(api_key=self.api_key or os.environ.get("GEMINI_API_KEY"))

                        config = types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            temperature=0.0
                        )
                        response = client.models.generate_content(
                            model=self.model_name,
                            contents=user_prompt,
                            config=config
                        )
                        return response.text

                    except ImportError:
                        import google.generativeai as genai
                        genai.configure(api_key=self.api_key or os.environ.get("GEMINI_API_KEY"))
                        model = genai.GenerativeModel(
                            model_name=self.model_name,
                            system_instruction=system_prompt,
                            generation_config={"response_mime_type": "application/json", "temperature": 0.0}
                        )
                        response = model.generate_content(user_prompt)
                        return response.text

                # 3. Anthropic Claude
                elif "claude" in self.model_name.lower():
                    import anthropic
                    client = anthropic.Anthropic(api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"))
                    response = client.messages.create(
                        model=self.model_name,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                        temperature=0.0,
                        max_tokens=1024
                    )
                    return response.content[0].text

                else:
                    raise ValueError(f"Unsupported judge model: {self.model_name}. Please provide a `custom_caller`.")

            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Judge API Call permanently failed after {self.max_retries} attempts: {e}")
                    raise
                err_str = str(e)
                wait_time = (attempt + 1) * 3
                if "429" in err_str or "quota" in err_str.lower() or "resourceexhausted" in err_str.lower():
                    match = re.search(r"retry in ([0-9.]+)s", err_str)
                    if not match:
                        match = re.search(r"seconds:\s*([0-9]+)", err_str)
                    if match:
                        wait_time = float(match.group(1)) + 2.0
                    else:
                        wait_time = max(wait_time, 10.0 * (attempt + 1))
                logger.warning(f"Judge API Call attempt {attempt + 1} failed ({e}). Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

    def _extract_json(self, raw_output: str) -> Dict[str, Any]:
        """
        Safely extracts JSON object from markdown blocks or raw text.
        """
        try:
            return json.loads(raw_output.strip())
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        match = re.search(r"(\{.*\})", raw_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse JSON from judge output:\n{raw_output}")
        return {"error": "Failed to parse JSON response", "raw_output": raw_output}

    def evaluate_single_response(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Evaluates a single model response across all 3 criteria:
        - Coherence & Sensicality (1-5)
        - Helpfulness & Factuality (1-5)
        - Hinglish Naturalness (1-5)

        Returns:
            Dict[str, Any]: Parsed JSON containing scores and justifications.
        """
        user_prompt = USER_PROMPT_SINGLE_EVAL.format(prompt=prompt, response=response)
        raw_output = self._call_judge_llm(SYSTEM_PROMPT_SINGLE_EVAL, user_prompt)
        return self._extract_json(raw_output)

    def evaluate_pairwise(self, prompt: str, response_a: str, response_b: str) -> Dict[str, Any]:
        """
        Runs a Blind Head-to-Head Pairwise comparison between Model A and Model B.

        Returns:
            Dict[str, Any]: Winner ('Model A', 'Model B', 'Tie'), critiques, and reasoning.
        """
        user_prompt = USER_PROMPT_PAIRWISE_EVAL.format(prompt=prompt, response_a=response_a, response_b=response_b)
        raw_output = self._call_judge_llm(SYSTEM_PROMPT_PAIRWISE_EVAL, user_prompt)
        return self._extract_json(raw_output)

    def evaluate_benchmark(
        self,
        test_samples: List[Dict[str, str]],
        max_workers: int = 8
    ) -> Dict[str, Any]:
        """
        Evaluates a benchmark dataset containing prompts, Model A (Base) responses,
        and Model B (LoRA) responses using parallel worker threads.

        Args:
            test_samples (List[Dict[str, str]]): List of dicts with keys:
                - 'prompt': user input
                - 'response_base': base model output
                - 'response_lora': fine-tuned model output
            max_workers (int): Number of concurrent worker threads.

        Returns:
            Dict[str, Any]: Comprehensive evaluation report including Win Rates, Mean Scores, and Breakdown.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        total = len(test_samples)
        logger.info(f"Running LLM Judge Evaluation across {total} samples with {max_workers} threads...")
        print(f"[LLM Judge] Running evaluation across {total} samples with {max_workers} parallel workers...", flush=True)

        model_a_scores = {"coherence": [], "helpfulness": [], "hinglish_naturalness": [], "overall": []}
        model_b_scores = {"coherence": [], "helpfulness": [], "hinglish_naturalness": [], "overall": []}
        wins = {"Model A": 0, "Model B": 0, "Tie": 0}

        def process_sample(item):
            idx, sample = item
            prompt = sample["prompt"]
            resp_base = sample.get("response_base", "")
            resp_lora = sample.get("response_lora", "")

            eval_base = self.evaluate_single_response(prompt, resp_base)
            eval_lora = self.evaluate_single_response(prompt, resp_lora)
            pairwise_res = self.evaluate_pairwise(prompt, resp_base, resp_lora)

            return idx, prompt, resp_base, resp_lora, eval_base, eval_lora, pairwise_res

        completed = 0
        results_by_idx = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_sample, (i, s)): i for i, s in enumerate(test_samples, 1)}
            for future in as_completed(futures):
                try:
                    idx, prompt, resp_base, resp_lora, eval_base, eval_lora, pairwise_res = future.result()
                    results_by_idx[idx] = (prompt, resp_base, resp_lora, eval_base, eval_lora, pairwise_res)
                    completed += 1
                    if completed % 5 == 0 or completed == 1 or completed == total:
                        print(f"[LLM Judge Progress] Completed {completed}/{total} samples...", flush=True)
                except Exception as e:
                    logger.error(f"Sample failed during parallel eval: {e}")
                    completed += 1

        # Aggregate sorted results
        detailed_results = []
        for i in range(1, total + 1):
            if i in results_by_idx:
                prompt, resp_base, resp_lora, eval_base, eval_lora, pairwise_res = results_by_idx[i]

                # Record scores for Base
                if "coherence" in eval_base and isinstance(eval_base["coherence"], dict) and "score" in eval_base["coherence"]:
                    model_a_scores["coherence"].append(eval_base["coherence"]["score"])
                    model_a_scores["helpfulness"].append(eval_base["helpfulness"]["score"])
                    model_a_scores["hinglish_naturalness"].append(eval_base["hinglish_naturalness"]["score"])
                    model_a_scores["overall"].append(eval_base.get("overall_weighted_score", 0))

                # Record scores for LoRA
                if "coherence" in eval_lora and isinstance(eval_lora["coherence"], dict) and "score" in eval_lora["coherence"]:
                    model_b_scores["coherence"].append(eval_lora["coherence"]["score"])
                    model_b_scores["helpfulness"].append(eval_lora["helpfulness"]["score"])
                    model_b_scores["hinglish_naturalness"].append(eval_lora["hinglish_naturalness"]["score"])
                    model_b_scores["overall"].append(eval_lora.get("overall_weighted_score", 0))

                winner = pairwise_res.get("winner", "Tie")
                wins[winner] = wins.get(winner, 0) + 1

                detailed_results.append({
                    "sample_id": i,
                    "prompt": prompt,
                    "base_model": {"response": resp_base, "evaluation": eval_base},
                    "lora_model": {"response": resp_lora, "evaluation": eval_lora},
                    "pairwise": pairwise_res
                })

        total = len(test_samples)
        
        def avg(lst): return round(sum(lst) / len(lst), 2) if lst else 0.0

        summary = {
            "total_evaluated": total,
            "win_rate_summary": {
                "lora_model_wins (Model B)": wins.get("Model B", 0),
                "lora_model_win_rate_pct": round((wins.get("Model B", 0) / total) * 100, 2) if total > 0 else 0,
                "base_model_wins (Model A)": wins.get("Model A", 0),
                "base_model_win_rate_pct": round((wins.get("Model A", 0) / total) * 100, 2) if total > 0 else 0,
                "ties": wins.get("Tie", 0),
                "tie_rate_pct": round((wins.get("Tie", 0) / total) * 100, 2) if total > 0 else 0
            },
            "mean_scores": {
                "base_model": {
                    "coherence": avg(model_a_scores["coherence"]),
                    "helpfulness": avg(model_a_scores["helpfulness"]),
                    "hinglish_naturalness": avg(model_a_scores["hinglish_naturalness"]),
                    "overall": avg(model_a_scores["overall"])
                },
                "lora_model": {
                    "coherence": avg(model_b_scores["coherence"]),
                    "helpfulness": avg(model_b_scores["helpfulness"]),
                    "hinglish_naturalness": avg(model_b_scores["hinglish_naturalness"]),
                    "overall": avg(model_b_scores["overall"])
                }
            },
            "detailed_samples": detailed_results
        }

        return summary
