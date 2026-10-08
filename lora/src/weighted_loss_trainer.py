import torch
import torch.nn as nn
from typing import Dict, Any, List
from transformers import Trainer, PreTrainedTokenizer

class ProgressiveWeightedLossCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, weights: Dict[str, float]):
        self.tokenizer = tokenizer
        self.weights = weights
        self.inst_end_tokens = self.tokenizer.encode("[/INST]", add_special_tokens=False)
        self.think_tokens = self.tokenizer.encode("<think>", add_special_tokens=False)
        self.end_think_tokens = self.tokenizer.encode("</think>", add_special_tokens=False)

    def compute_weight_mask(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        weight_mask = torch.zeros_like(input_ids, dtype=torch.float32)

        for b in range(batch_size):
            tokens = input_ids[b].tolist()
            # Default state: prompt
            state = "prompt"
            i = 0
            while i < seq_len:
                if state == "prompt":
                    weight_mask[b, i] = self.weights.get("prompt", 0.0)
                    if i + len(self.inst_end_tokens) <= seq_len and tokens[i:i+len(self.inst_end_tokens)] == self.inst_end_tokens:
                        weight_mask[b, i:i+len(self.inst_end_tokens)] = self.weights.get("prompt", 0.0)
                        i += len(self.inst_end_tokens)
                        state = "before_think"
                        continue
                elif state == "before_think":
                    if i + len(self.think_tokens) <= seq_len and tokens[i:i+len(self.think_tokens)] == self.think_tokens:
                        weight_mask[b, i:i+len(self.think_tokens)] = self.weights.get("think_tag_weight", 0.25)
                        i += len(self.think_tokens)
                        state = "reasoning"
                        continue
                    else:
                        weight_mask[b, i] = self.weights.get("answer_weight", 1.0)
                elif state == "reasoning":
                    if i + len(self.end_think_tokens) <= seq_len and tokens[i:i+len(self.end_think_tokens)] == self.end_think_tokens:
                        weight_mask[b, i:i+len(self.end_think_tokens)] = self.weights.get("end_think_tag_weight", 0.75)
                        i += len(self.end_think_tokens)
                        state = "answer"
                        continue
                    else:
                        weight_mask[b, i] = self.weights.get("reasoning_weight", 0.50)
                elif state == "answer":
                    weight_mask[b, i] = self.weights.get("answer_weight", 1.0)
                i += 1

        return weight_mask

class ProgressiveWeightedLossTrainer(Trainer):
    def __init__(self, *args, weight_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_collator = weight_collator

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Shift logits and labels for causal LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        weight_mask = self.weight_collator.compute_weight_mask(inputs["input_ids"], labels)
        shift_weights = weight_mask[..., 1:].contiguous().to(logits.device)

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        weighted_loss = loss * shift_weights.view(-1)
        active_weights = shift_weights.view(-1).sum()
        final_loss = weighted_loss.sum() / (active_weights + 1e-8)

        return (final_loss, outputs) if return_outputs else final_loss
