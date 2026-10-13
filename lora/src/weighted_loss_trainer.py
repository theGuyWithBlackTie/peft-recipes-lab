import inspect
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from transformers import PreTrainedTokenizer

try:
    from trl import SFTTrainer
    BaseTrainer = SFTTrainer
except ImportError:
    from transformers import Trainer
    BaseTrainer = Trainer

class ProgressiveWeightedLossCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, weights: Dict[str, float]):
        self.tokenizer = tokenizer
        self.weights = weights
        
        # Candidate token sequences to handle leading spaces/newlines across tokenizers
        self.inst_candidates = [
            self.tokenizer.encode("[/INST]", add_special_tokens=False),
            self.tokenizer.encode(" [/INST]", add_special_tokens=False),
        ]
        self.think_candidates = [
            self.tokenizer.encode("<think>", add_special_tokens=False),
            self.tokenizer.encode(" <think>", add_special_tokens=False),
            self.tokenizer.encode("\n<think>", add_special_tokens=False),
        ]
        self.end_think_candidates = [
            self.tokenizer.encode("</think>", add_special_tokens=False),
            self.tokenizer.encode("\n</think>", add_special_tokens=False),
            self.tokenizer.encode(" </think>", add_special_tokens=False),
        ]

    def _match_candidate(self, tokens: List[int], start_idx: int, candidates: List[List[int]]) -> Optional[int]:
        """Returns the length of the matching candidate sequence, or None."""
        seq_len = len(tokens)
        for cand in candidates:
            if not cand:
                continue
            cand_len = len(cand)
            if start_idx + cand_len <= seq_len and tokens[start_idx:start_idx+cand_len] == cand:
                return cand_len
        return None

    def compute_weight_mask(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        weight_mask = torch.zeros_like(input_ids, dtype=torch.float32)

        for b in range(batch_size):
            tokens = input_ids[b].tolist()
            state = "prompt"
            i = 0
            while i < seq_len:
                if state == "prompt":
                    weight_mask[b, i] = self.weights.get("prompt", 0.0)
                    match_len = self._match_candidate(tokens, i, self.inst_candidates)
                    if match_len is not None:
                        weight_mask[b, i:i+match_len] = self.weights.get("prompt", 0.0)
                        i += match_len
                        state = "before_think"
                        continue
                elif state == "before_think":
                    match_len = self._match_candidate(tokens, i, self.think_candidates)
                    if match_len is not None:
                        weight_mask[b, i:i+match_len] = self.weights.get("think_tag_weight", 0.25)
                        i += match_len
                        state = "reasoning"
                        continue
                    else:
                        weight_mask[b, i] = self.weights.get("answer_weight", 1.0)
                elif state == "reasoning":
                    match_len = self._match_candidate(tokens, i, self.end_think_candidates)
                    if match_len is not None:
                        weight_mask[b, i:i+match_len] = self.weights.get("end_think_tag_weight", 0.75)
                        i += match_len
                        state = "answer"
                        continue
                    else:
                        weight_mask[b, i] = self.weights.get("reasoning_weight", 0.50)
                elif state == "answer":
                    weight_mask[b, i] = self.weights.get("answer_weight", 1.0)
                i += 1

        return weight_mask

class ProgressiveWeightedLossTrainer(BaseTrainer):
    def __init__(self, *args, weight_collator: Optional[ProgressiveWeightedLossCollator] = None, **kwargs):
        # Inspect parent class parameters to safely filter out unexpected arguments
        parent_params = inspect.signature(super().__init__).parameters
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parent_params.values())

        filtered_kwargs = dict(kwargs)
        if filtered_kwargs.get("peft_config") is None:
            filtered_kwargs.pop("peft_config", None)

        if not accepts_var_kwargs:
            filtered_kwargs = {k: v for k, v in filtered_kwargs.items() if k in parent_params}

        super().__init__(*args, **filtered_kwargs)
        self.weight_collator = weight_collator

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is None:
            labels = inputs["input_ids"].clone()

        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Shift logits and labels for causal LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        weight_mask = self.weight_collator.compute_weight_mask(inputs["input_ids"], labels)
        shift_weights = weight_mask[..., 1:].contiguous().to(logits.device)

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Only compute loss on active tokens where label != -100 (ignores padding and masked prompt)
        active_mask = (shift_labels.view(-1) != -100).float()
        flat_weights = shift_weights.view(-1) * active_mask
        weighted_loss = loss * flat_weights
        active_weights = flat_weights.sum()
        final_loss = weighted_loss.sum() / (active_weights + 1e-8)

        return (final_loss, outputs) if return_outputs else final_loss
