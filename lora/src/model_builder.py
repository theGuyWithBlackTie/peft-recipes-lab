import torch
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    AutoTokenizer,
    PreTrainedTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)

from typing import Tuple
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaModelBuilder:
    """
    A builder class responsible for instantiating and configuring a LLama model and its tokenizer for LoRA fine-tuning.
    """
    def __init__(self, config: dict):
        """
        Initializes the builder with the provided configuration.
        
        Args:
            config (dict): The configuration dictionary (usually loaded from a YAML file)
        """
        self.config = config
        self.model_id = self.config.get("model_id", "meta-llama/Llama-3.2-1B-Instruct")

        # Initialize internal state
        self.tokenizer = None
        self.model = None
    
    def _setup_tokenizer(self) -> PreTrainedTokenizer:
        """
        Loads and applies Llama-3 specific tokenizer configurations."""
        logger.info(f"Loading tokenizer for model: {self.model_id}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Llama-3 requires mapping the pad token to the eos token for training
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        return tokenizer
    
    def _load_base_model(self) -> PreTrainedModel:
        """
        Loads the base language model in standard precision (bf16/fp16/fp32) for LoRA fine-tuning.
        Supports both single-GPU and multi-GPU DDP under accelerate launch.
        """
        torch_dtype = (
            torch.bfloat16 if self.config.get("use_bf16", True)
            else (torch.float16 if self.config.get("use_fp16", False) else torch.float32)
        )
        
        # In DDP (accelerate launch), each process binds to its specific LOCAL_RANK GPU
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            device_map = {"": local_rank}
            logger.info(f"DDP Distributed mode detected: binding to GPU {local_rank}")
        else:
            device_map = self.config.get("device_map", "auto")

        logger.info(f"Loading base model {self.model_id} with torch_dtype={torch_dtype}, device_map={device_map}...")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="sdpa",
            trust_remote_code=True,
            use_cache=False
        )

        return model

    def _apply_lora(self, model: PreTrainedModel) -> PeftModel:
        """
        Injects trainable LoRA adapters into the frozen base model.
        """
        target_modules = self.config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])

        logger.info("Initializing LoRA adapters...")
        peft_config = LoraConfig(
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )

        return get_peft_model(model, peft_config)

    def build(self) -> Tuple[PeftModel, PreTrainedTokenizer]:
        """
        Orchestrates the build pipeline.

        Returns:
            Tuple[PeftModel, PreTrainedTokenizer]: The fully configured model and tokenizer.
        """
        self.tokenizer = self._setup_tokenizer()
        base_model = self._load_base_model()
        self.model = self._apply_lora(base_model)

        # Verify setup
        self.model.print_trainable_parameters()
        logger.info("Model and tokenizer successfully built.")

        return self.model, self.tokenizer