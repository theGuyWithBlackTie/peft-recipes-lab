import os
import torch
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)

from typing import Tuple
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelBuilder:
    """
    A builder class responsible for instantiating and configuring any CausalLM model (Mistral, etc.)
    and tokenizer for LoRA / QLoRA fine-tuning.
    """
    def __init__(self, config: dict):
        """
        Initializes the builder with the provided configuration.
        
        Args:
            config (dict): The configuration dictionary (usually loaded from a YAML file)
        """
        self.config = config
        self.model_id = self.config.get("model_id", "mistralai/Mistral-7B-Instruct-v0.3")

        # Initialize internal state
        self.tokenizer = None
        self.model = None
    
    def _setup_tokenizer(self) -> PreTrainedTokenizer:
        """
        Loads and applies model-specific tokenizer configurations.
        """
        logger.info(f"Loading tokenizer for model: {self.model_id}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        # Set pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        tokenizer.padding_side = "right"

        return tokenizer
    
    def _load_base_model(self) -> PreTrainedModel:
        """
        Loads the base language model in standard precision (fp16/bf16) or 4-bit QLoRA.
        Dynamically configures precision based on hardware (e.g. FP16 on Kaggle P100).
        """
        native_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
        requested_bf16 = self.config.get("use_bf16", False) and native_bf16
        requested_fp16 = self.config.get("use_fp16", True)

        torch_dtype = (
            torch.bfloat16 if requested_bf16
            else (torch.float16 if (requested_fp16 and torch.cuda.is_available()) else torch.float32)
        )
        
        # In DDP (accelerate launch), device_map MUST be None so Trainer handles device placement
        if "LOCAL_RANK" in os.environ or "WORLD_SIZE" in os.environ:
            device_map = None
            logger.info("DDP Distributed mode detected: letting Trainer manage GPU device placement (device_map=None)")
        else:
            device_map = self.config.get("device_map", "auto")

        use_4bit = self.config.get("use_4bit", True)
        quantization_config = None

        if use_4bit:
            logger.info(f"Configuring 4-bit QLoRA (NF4) for {self.model_id} (Compute Dtype: {torch_dtype})...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        attn_impl = self.config.get("attn_implementation", "sdpa" if torch.cuda.is_available() else "eager")

        logger.info(f"Loading base model {self.model_id} with torch_dtype={torch_dtype}, use_4bit={use_4bit}, device_map={device_map}...")

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
            "use_cache": False,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        except Exception as e:
            if "attn_implementation" in model_kwargs:
                logger.warning(f"Attn implementation '{attn_impl}' failed ({e}). Retrying with fallback attention...")
                model_kwargs.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            else:
                raise e

        if use_4bit:
            use_grad_ckpt = self.config.get("gradient_checkpointing", True)
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_grad_ckpt)

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