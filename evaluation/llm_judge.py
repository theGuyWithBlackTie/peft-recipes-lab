import os
import json
import re
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
        model_name: str = "gpt-4o-mini",
        custom_caller: Optional[Callable[[str, str], str]] = None,
        api_key: Optional[str] = None
    ):
        """
        Initializes the LLM Judge.

        Args:
            model_name (str): Judge model identifier (e.g. 'gpt-4o-mini', 'gemini-1.5-flash', 'claude-3-5-sonnet-20241022').
            custom_caller (Callable, optional): Custom function taking (system_prompt, user_prompt) -> response_str.
            api_key (str, optional): API key for the selected provider.
        """
        self.model_name = model_name
        self.custom_caller = custom_caller
        self.api_key = api_key

    def _call_judge_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Routes system prompt and user input to the configured LLM backend.
        """
        if self.custom_caller is not None:
            return self.custom_caller(system_prompt, user_prompt)

        # 1. OpenAI / Compatible endpoint
        if "gpt" in self.model_name.lower() or "o1" in self.model_name.lower():
            try:
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
            except Exception as e:
                logger.error(f"OpenAI Judge Call Error: {e}")
                raise

        # 2. Google Gemini
        elif "gemini" in self.model_name.lower():
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key or os.environ.get("GEMINI_API_KEY"))
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_prompt,
                    generation_config={"response_mime_type": "application/json", "temperature": 0.0}
                )
                response = model.generate_content(user_prompt)
                return response.text
            except Exception as e:
                logger.error(f"Gemini Judge Call Error: {e}")
                raise

        # 3. Anthropic Claude
        elif "claude" in self.model_name.lower():
            try:
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
            except Exception as e:
                logger.error(f"Anthropic Judge Call Error: {e}")
                raise

        else:
            raise ValueError(f"Unsupported judge model: {self.model_name}. Please provide a `custom_caller`.")

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
        test_samples: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Evaluates a benchmark dataset containing prompts, Model A (Base) responses,
        and Model B (LoRA) responses.

        Args:
            test_samples (List[Dict[str, str]]): List of dicts with keys:
                - 'prompt': user input
                - 'response_base': base model output
                - 'response_lora': fine-tuned model output

        Returns:
            Dict[str, Any]: Comprehensive evaluation report including Win Rates, Mean Scores, and Breakdown.
        """
        logger.info(f"Running LLM Judge Evaluation across {len(test_samples)} benchmark samples...")

        model_a_scores = {"coherence": [], "helpfulness": [], "hinglish_naturalness": [], "overall": []}
        model_b_scores = {"coherence": [], "helpfulness": [], "hinglish_naturalness": [], "overall": []}
        
        wins = {"Model A": 0, "Model B": 0, "Tie": 0}
        detailed_results = []

        for i, sample in enumerate(test_samples, 1):
            prompt = sample["prompt"]
            resp_base = sample.get("response_base", "")
            resp_lora = sample.get("response_lora", "")

            # 1. Single response evaluations
            eval_base = self.evaluate_single_response(prompt, resp_base)
            eval_lora = self.evaluate_single_response(prompt, resp_lora)

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

            # 2. Pairwise Blind comparison
            pairwise_res = self.evaluate_pairwise(prompt, resp_base, resp_lora)
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
