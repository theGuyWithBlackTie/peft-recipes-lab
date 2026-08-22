import math
import logging
from typing import Dict, Any, Optional, Union, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerplexityEvaluator:
    """
    Evaluator for computing quantitative language modeling metrics:
    - Cross-Entropy Loss (nats/token)
    - Perplexity (exp(Loss))
    - Relative Perplexity Reduction (%)
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initializes the Perplexity Evaluator.

        Args:
            device (str, optional): Target device ('cuda', 'cpu', etc.).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def compute_perplexity(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Union[Dataset, List[str]],
        text_column: str = "text",
        max_samples: int = 100,
        max_seq_length: int = 512,
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """
        Computes Cross-Entropy Loss and Perplexity over a dataset.

        Args:
            model (PreTrainedModel): Language model to evaluate.
            tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the model.
            dataset (Dataset or List[str]): Evaluation dataset or list of texts.
            text_column (str): Name of text column if dataset is a Hugging Face Dataset.
            max_samples (int): Maximum number of samples to evaluate for speed.
            max_seq_length (int): Maximum sequence length for truncation.
            batch_size (int): Evaluation batch size.

        Returns:
            Dict[str, Any]: Dictionary containing loss, perplexity, and evaluated sample count.
        """
        model.eval()
        device = next(model.parameters()).device

        # Extract text list
        if isinstance(dataset, Dataset):
            num_eval = min(len(dataset), max_samples)
            texts = [dataset[i][text_column] for i in range(num_eval)]
        else:
            num_eval = min(len(dataset), max_samples)
            texts = dataset[:num_eval]

        logger.info(f"Evaluating Perplexity on {len(texts)} samples (max_seq_length={max_seq_length})...")

        total_loss = 0.0
        total_tokens = 0

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                
                encodings = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length
                ).to(device)

                input_ids = encodings["input_ids"]
                attention_mask = encodings["attention_mask"]

                # Labels: set pad tokens to -100 so loss is ignored for padding
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                # Number of active non-pad tokens in this batch
                batch_tokens = (labels != -100).sum().item()
                
                if batch_tokens > 0:
                    total_loss += outputs.loss.item() * batch_tokens
                    total_tokens += batch_tokens

        if total_tokens == 0:
            return {"loss": 0.0, "perplexity": 0.0, "total_tokens": 0, "sample_count": 0}

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return {
            "loss": round(avg_loss, 4),
            "perplexity": round(perplexity, 2),
            "total_tokens_evaluated": total_tokens,
            "sample_count": len(texts)
        }

    def compare_models(
        self,
        base_model: PreTrainedModel,
        lora_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Union[Dataset, List[str]],
        max_samples: int = 100,
        max_seq_length: int = 512
    ) -> Dict[str, Any]:
        """
        Calculates and compares Perplexity between Base Model (Before) and LoRA Model (After).

        Returns:
            Dict[str, Any]: Comparative statistics including perplexity reduction percentage.
        """
        logger.info("Computing Baseline Perplexity on Base Model (Before)...")
        base_res = self.compute_perplexity(
            base_model, tokenizer, dataset, max_samples=max_samples, max_seq_length=max_seq_length
        )

        logger.info("Computing Perplexity on LoRA Fine-Tuned Model (After)...")
        lora_res = self.compute_perplexity(
            lora_model, tokenizer, dataset, max_samples=max_samples, max_seq_length=max_seq_length
        )

        base_ppl = base_res["perplexity"]
        lora_ppl = lora_res["perplexity"]
        
        # Calculate perplexity drop percentage
        ppl_reduction = ((base_ppl - lora_ppl) / base_ppl) * 100 if base_ppl > 0 else 0.0

        return {
            "base_model": base_res,
            "lora_model": lora_res,
            "perplexity_reduction_pct": round(ppl_reduction, 2),
            "improvement_summary": f"Perplexity improved from {base_ppl:.2f} to {lora_ppl:.2f} ({ppl_reduction:.1f}% reduction in uncertainty)"
        }
