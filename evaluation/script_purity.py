import re
import unicodedata
from typing import Dict, Any, List, Union


class ScriptPurityEvaluator:
    """
    Evaluator for analyzing character-level and token-level script purity.
    Ensures that generated Hinglish text strictly adheres to Romanized (Latin) script
    and flags any accidental Devanagari or unsupported character bleed.
    """

    def __init__(self, target_script: str = "LATIN"):
        """
        Initializes the script purity evaluator.

        Args:
            target_script (str): Target script name (default: "LATIN").
        """
        self.target_script = target_script.upper()
        # Unicode range for Devanagari: U+0900 to U+097F
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        # Unicode range for Latin characters: a-z, A-Z (and extended Latin accents)
        self.latin_pattern = re.compile(r'[a-zA-Z\u00C0-\u024F]')

    def evaluate_text(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a single text string for script purity.

        Args:
            text (str): The text to evaluate.

        Returns:
            Dict[str, Any]: Detailed character counts, ratios, and purity verdict.
        """
        if not text or not text.strip():
            return {
                "total_chars": 0,
                "total_letters": 0,
                "latin_chars": 0,
                "devanagari_chars": 0,
                "other_chars": 0,
                "latin_purity_score": 1.0,
                "devanagari_ratio": 0.0,
                "is_pure_romanized": True,
                "detected_scripts": []
            }

        total_chars = len(text)
        latin_chars = len(self.latin_pattern.findall(text))
        devanagari_chars = len(self.devanagari_pattern.findall(text))
        
        # Calculate letter-only counts
        letters = [c for c in text if c.isalpha()]
        total_letters = len(letters)
        
        detected_scripts = set()
        for char in letters:
            try:
                script_name = unicodedata.name(char).split()[0]
                detected_scripts.add(script_name)
            except ValueError:
                pass

        other_chars = total_letters - (latin_chars + devanagari_chars)

        # Latin purity score calculated against alphabetical letters
        latin_purity_score = (latin_chars / total_letters) if total_letters > 0 else 1.0
        devanagari_ratio = (devanagari_chars / total_letters) if total_letters > 0 else 0.0

        # Text is considered pure Romanized if >= 99% of letters are Latin
        is_pure_romanized = latin_purity_score >= 0.99 and devanagari_chars == 0

        return {
            "total_chars": total_chars,
            "total_letters": total_letters,
            "latin_chars": latin_chars,
            "devanagari_chars": devanagari_chars,
            "other_chars": other_chars,
            "latin_purity_score": round(latin_purity_score * 100, 2),  # In percentage (0 - 100%)
            "devanagari_ratio": round(devanagari_ratio * 100, 2),       # In percentage (0 - 100%)
            "is_pure_romanized": is_pure_romanized,
            "detected_scripts": sorted(list(detected_scripts))
        }

    def evaluate_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        Evaluates a batch of generated texts and aggregates script purity metrics.

        Args:
            texts (List[str]): List of generated responses.

        Returns:
            Dict[str, Any]: Aggregated summary metrics.
        """
        if not texts:
            return {"sample_count": 0, "avg_latin_purity": 100.0, "pure_romanized_rate": 100.0}

        results = [self.evaluate_text(t) for t in texts]
        total_samples = len(results)
        pure_count = sum(1 for r in results if r["is_pure_romanized"])
        avg_purity = sum(r["latin_purity_score"] for r in results) / total_samples
        total_devanagari_chars = sum(r["devanagari_chars"] for r in results)

        return {
            "sample_count": total_samples,
            "pure_romanized_samples": pure_count,
            "pure_romanized_rate": round((pure_count / total_samples) * 100, 2),
            "avg_latin_purity_score": round(avg_purity, 2),
            "total_devanagari_violations": total_devanagari_chars,
            "per_sample_results": results
        }
