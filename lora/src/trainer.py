import os
import logging
from typing import Dict, Any, Optional

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from datasets import Dataset
from trl import SFTTrainer

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaTrainer:
    """
    A class that encapsulates the setup and execution of the fine-tuning loop
    using Hugging Face's SFTTrainer for LLama-3 LoRA.
    """

    def __init__(self, 
        config: dict, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        train_dataset: Dataset, 
        val_dataset: Optional[Dataset] = None
    ):
        """
        Initializes the trainer wrapper.

        Args:
            config (dict): The configuration dictionary (usually loaded from a YAML file)
            model (PreTrainedModel): The PEFT-wrapped LLaMA model ready for LoRA tuning
            tokenizer (PreTrainedTokenizer): The LLaMA tokenizer
            train_dataset (Dataset): The training dataset
            val_dataset (Dataset): The validation dataset
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.output_dir = self.config.get("output_dir", "./llama3-lora-outputs")

    def _get_training_arguments(self) -> Any:
        """
        Maps configuration variables to Hugging Face TrainingArguments or SFTConfig.
        Defaults are optimized for LoRA fine-tuning.
        """
        import inspect
        logger.info("Configuring Training Arguments...")

        # 1. Check if SFTConfig is available (trl >= 0.12.0)
        try:
            from trl import SFTConfig
            training_args_cls = SFTConfig
        except ImportError:
            training_args_cls = TrainingArguments

        # evaluation strategy must match eval_steps logic
        do_eval = self.val_dataset is not None
        eval_strategy = "steps" if do_eval else "no"
        
        eval_steps = self.config.get("eval_steps")
        if eval_steps is None:
            eval_steps = 500 if do_eval else 0

        # Build candidate arguments dictionary
        candidate_args = {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": self.config.get("batch_size", 4),
            "per_device_eval_batch_size": self.config.get("batch_size", 4),
            "gradient_accumulation_steps": self.config.get("gradient_accumulation_steps", 4),
            "optim": self.config.get("optim", "adamw_torch"),
            "learning_rate": float(self.config.get("learning_rate", 2e-4)),
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "max_grad_norm": 0.3,
            "num_train_epochs": self.config.get("epochs", 1),
            "logging_steps": self.config.get("logging_steps", 10),
            "save_strategy": "steps",
            "save_steps": self.config.get("save_steps", 50),
            "bf16": self.config.get("use_bf16", True),
            "fp16": self.config.get("use_fp16", False),
            "report_to": "none",
            # SFTConfig specific parameters (trl >= 0.12.0)
            "dataset_text_field": "text",
            "max_seq_length": self.config.get("max_seq_length", 2048),
            "packing": False,
        }

        # Inspect valid parameters for the target config class
        valid_params = inspect.signature(training_args_cls.__init__).parameters

        # Handle eval strategy naming differences across transformers versions
        if "eval_strategy" in valid_params:
            candidate_args["eval_strategy"] = eval_strategy
        elif "evaluation_strategy" in valid_params:
            candidate_args["evaluation_strategy"] = eval_strategy

        if do_eval and eval_steps > 0 and "eval_steps" in valid_params:
            candidate_args["eval_steps"] = eval_steps

        # Filter strictly to arguments accepted by the installed class
        filtered_args = {k: v for k, v in candidate_args.items() if k in valid_params}
        training_args = training_args_cls(**filtered_args)

        return training_args

    def _setup_trainer(self, training_args: Any) -> SFTTrainer:
        """
        Instantiates the Supervised Fine-Tuning (SFT) Trainer dynamically
        compatible with both newer (SFTConfig/processing_class) and older TRL releases.
        """
        import inspect
        logger.info("Initializing SFTTrainer...")

        sft_init_params = inspect.signature(SFTTrainer.__init__).parameters

        trainer_kwargs = {
            "model": self.model,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.val_dataset,
            "args": training_args,
        }

        # Handle tokenizer vs processing_class across transformers / TRL versions
        if "processing_class" in sft_init_params:
            trainer_kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in sft_init_params:
            trainer_kwargs["tokenizer"] = self.tokenizer

        # Legacy TRL parameter support (passed directly to SFTTrainer in older versions)
        if "dataset_text_field" in sft_init_params:
            trainer_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in sft_init_params:
            trainer_kwargs["max_seq_length"] = self.config.get("max_seq_length", 2048)
        if "packing" in sft_init_params:
            trainer_kwargs["packing"] = False
        if "peft_config" in sft_init_params:
            trainer_kwargs["peft_config"] = None

        return SFTTrainer(**trainer_kwargs)

    def train_and_save(self) -> None:
        """
        Executes the training loop and saves the final adapters.
        """
        training_args = self._get_training_arguments()
        trainer = self._setup_trainer(training_args)

        logger.info("Starting training loop...")
        trainer.train()

        logger.info(f"Training complete. Saving final model adapters to {self.output_dir}...")
        trainer.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        logger.info("Save successful. Fine-tuning complete!")


# Convenience wrapper for the main script
def run_training(
    config: dict, 
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    train_data: Dataset, 
    val_data: Optional[Dataset] = None
) -> None:
    """Convenience wrapper for the LlamaTrainer class."""
    trainer = LlamaTrainer(config, model, tokenizer, train_data, val_data)
    trainer.train_and_save()

if __name__ == "__main__":
    # Test logic
    pass
        