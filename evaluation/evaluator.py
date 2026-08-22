import os
import json
import logging
from typing import Dict, Any, List, Optional, Union

from evaluation.perplexity import PerplexityEvaluator
from evaluation.script_purity import ScriptPurityEvaluator
from evaluation.llm_judge import LLMJudge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HinglishEvaluator:
    """
    Unified Evaluation Suite for Hinglish Language Models.
    Combines:
    1. Quantitative Perplexity (Cross-Entropy Loss / PPL)
    2. Character-level Script Purity (% Latin Script vs Devanagari)
    3. Multi-dimensional LLM-as-a-Judge (Coherence, Helpfulness, Hinglish Naturalness, Win Rate)
    """

    def __init__(
        self,
        judge_model_name: str = "gpt-4o-mini",
        judge_api_key: Optional[str] = None
    ):
        """
        Initializes the unified evaluator.
        """
        self.perplexity_evaluator = PerplexityEvaluator()
        self.script_evaluator = ScriptPurityEvaluator()
        self.judge = LLMJudge(model_name=judge_model_name, api_key=judge_api_key)

    def evaluate_generation_quality(
        self,
        test_samples: List[Dict[str, str]],
        run_llm_judge: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates a set of generated test samples.

        Args:
            test_samples (List[Dict[str, str]]): List of dicts with:
                - 'prompt': user input
                - 'response_base': (optional) base model output
                - 'response_lora': fine-tuned model output
            run_llm_judge (bool): Whether to trigger LLM-as-a-Judge evaluation.

        Returns:
            Dict[str, Any]: Consolidated evaluation report.
        """
        lora_responses = [s.get("response_lora", "") for s in test_samples if s.get("response_lora")]
        base_responses = [s.get("response_base", "") for s in test_samples if s.get("response_base")]

        # 1. Script Purity Evaluation
        logger.info("Evaluating Script Purity for generated responses...")
        lora_purity = self.script_evaluator.evaluate_batch(lora_responses)
        base_purity = self.script_evaluator.evaluate_batch(base_responses) if base_responses else None

        # 2. LLM-as-a-Judge Evaluation
        judge_report = None
        if run_llm_judge and base_responses and lora_responses:
            logger.info("Running LLM Judge Evaluation...")
            try:
                judge_report = self.judge.evaluate_benchmark(test_samples)
            except Exception as e:
                logger.warning(f"LLM Judge Evaluation skipped or encountered error: {e}")
                judge_report = {"error": str(e)}

        report = {
            "total_samples": len(test_samples),
            "script_purity": {
                "lora_model": lora_purity,
                "base_model": base_purity
            },
            "llm_judge_report": judge_report
        }

        return report

    def generate_markdown_summary(self, evaluation_report: Dict[str, Any]) -> str:
        """
        Formats the evaluation report into a clean, readable Markdown dashboard.
        """
        purity = evaluation_report.get("script_purity", {}).get("lora_model", {})
        judge = evaluation_report.get("llm_judge_report", {})
        means = judge.get("mean_scores", {})
        win_rates = judge.get("win_rate_summary", {})

        md = "# 🇮🇳 Hinglish Model Evaluation Report\n\n"
        
        # Section 1: Script Purity
        md += "## 1. 🔤 Script Purity (Romanized Latin Script Adherence)\n\n"
        md += f"* **Average Latin Purity:** `{purity.get('avg_latin_purity_score', 0)}%`\n"
        md += f"* **Pure Romanized Output Rate:** `{purity.get('pure_romanized_rate', 0)}%`\n"
        md += f"* **Devanagari Violations:** `{purity.get('total_devanagari_violations', 0)} characters`\n\n"

        # Section 2: LLM Judge
        if "mean_scores" in judge:
            md += "## 2. 🤖 LLM-as-a-Judge Scores (Scale: 1 to 5)\n\n"
            md += "| Evaluation Metric | Base Model (Before) | LoRA Fine-Tuned (After) | Gain |\n"
            md += "|---|:---:|:---:|:---:|\n"
            
            base_m = means.get("base_model", {})
            lora_m = means.get("lora_model", {})

            for metric in ["coherence", "helpfulness", "hinglish_naturalness", "overall"]:
                b_val = base_m.get(metric, 0.0)
                l_val = lora_m.get(metric, 0.0)
                diff = l_val - b_val
                sign = "+" if diff > 0 else ""
                md += f"| **{metric.replace('_', ' ').title()}** | {b_val:.2f} / 5.0 | **{l_val:.2f} / 5.0** | `{sign}{diff:.2f}` |\n"
            
            md += "\n"

        # Section 3: Head-to-Head Pairwise Win Rate
        if win_rates:
            md += "## 3. 🏆 Head-to-Head Pairwise Win Rate\n\n"
            md += f"* **LoRA Fine-Tuned Wins:** `{win_rates.get('lora_model_wins (Model B)', 0)}` ({win_rates.get('lora_model_win_rate_pct', 0)}%)\n"
            md += f"* **Base Model Wins:** `{win_rates.get('base_model_wins (Model A)', 0)}` ({win_rates.get('base_model_win_rate_pct', 0)}%)\n"
            md += f"* **Ties:** `{win_rates.get('ties', 0)}` ({win_rates.get('tie_rate_pct', 0)}%)\n\n"

        return md
