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
    prepare_model_for_kbit_training,
    PeftModel
)

from typing import Tuple
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaModelBuilder:
    """
    A builder class responsible for instantiating and configuring a LLama model and its tokenizer for (Q)Lora fine-tuning.
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
        Loads and applies Llama-3 specific tokenizer configurationns."""
        logger.info(f"Loading tokenizer for model: {self.model_id}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Llama-3 requires mapping the pad token to the eos token for training
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        return tokenizer
    
    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """
        Constructs the 4-bit configuration object"""
        logger.info("Configuring BitsAndBytes for 4-bit quantization")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    def _load_base_model(self) -> PreTrainedModel:
        """
        Loads the base language model with the specified quantization configuration and prepares it for k-bit training.
        """
        logger.info(f"Loading base mode {self.model_id} in 4-bit...")
        quant_config = self._get_quantization_config()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            device_map="auto",
            use_cache=False
        )

        # Prepare the model for PEFT (enables gradient checkpointing, layer norm casting)
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)

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