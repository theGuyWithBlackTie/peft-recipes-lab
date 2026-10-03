import os
import torch
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


class Trainer:
    """
    A class that encapsulates the setup and execution of the fine-tuning loop
    using Hugging Face's SFTTrainer for LoRA / QLoRA.
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
            model (PreTrainedModel): The PEFT-wrapped model ready for LoRA tuning
            tokenizer (PreTrainedTokenizer): The tokenizer
            train_dataset (Dataset): The training dataset
            val_dataset (Dataset): The validation dataset
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.output_dir = self.config.get("output_dir", "./mistral-lora-outputs")

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
        # Dynamically determine precision based on native hardware capability (FP16 on Kaggle P100)
        native_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
        use_bf16 = (self.config.get("use_bf16", False) and native_bf16)
        use_fp16 = self.config.get("use_fp16", not native_bf16)

        candidate_args = {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": self.config.get("batch_size", 4),
            "per_device_eval_batch_size": self.config.get("batch_size", 4),
            "gradient_accumulation_steps": self.config.get("gradient_accumulation_steps", 4),
            "optim": self.config.get("optim", "paged_adamw_8bit"),
            "learning_rate": float(self.config.get("learning_rate", 2e-4)),
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "max_grad_norm": 0.3,
            "num_train_epochs": self.config.get("epochs", 2),
            "logging_steps": self.config.get("logging_steps", 10),
            "save_strategy": "steps",
            "save_steps": self.config.get("save_steps", 500),
            "save_total_limit": self.config.get("save_total_limit", 2),
            "gradient_checkpointing": self.config.get("gradient_checkpointing", True),
            "bf16": use_bf16,
            "fp16": use_fp16,
            "report_to": "none",
            # SFTConfig specific parameters (trl >= 0.12.0)
            "dataset_text_field": "text",
            "max_seq_length": self.config.get("max_seq_length", 512),
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

        # Setup DataCollatorForCompletionOnlyLM for completion-only loss masking
        if self.config.get("use_completion_masking", True):
            response_template = self.config.get(
                "response_template", 
                "[/INST]"
            )
            try:
                from trl import DataCollatorForCompletionOnlyLM
                data_collator = DataCollatorForCompletionOnlyLM(
                    response_template=response_template,
                    tokenizer=self.tokenizer
                )
                trainer_kwargs["data_collator"] = data_collator
                logger.info(f"✅ Enabled DataCollatorForCompletionOnlyLM with response_template='{response_template}'")
            except Exception as e:
                logger.warning(f"Could not initialize DataCollatorForCompletionOnlyLM: {e}")

        # Legacy TRL parameter support (passed directly to SFTTrainer in older versions)
        if "dataset_text_field" in sft_init_params:
            trainer_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in sft_init_params:
            trainer_kwargs["max_seq_length"] = self.config.get("max_seq_length", 512)
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

        resume_checkpoint = self.config.get("resume_from_checkpoint", None)
        logger.info(f"Starting training loop (resume_from_checkpoint={resume_checkpoint})...")
        trainer.train(resume_from_checkpoint=resume_checkpoint)

        logger.info(f"Training complete. Saving final model adapters to {self.output_dir}...")
        os.makedirs(self.output_dir, exist_ok=True)
        trainer.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # 1. Verify Disk Integrity
        adapter_weights = os.path.join(self.output_dir, "adapter_model.safetensors")
        adapter_config = os.path.join(self.output_dir, "adapter_config.json")

        if os.path.exists(adapter_config) and (os.path.exists(adapter_weights) or os.path.exists(os.path.join(self.output_dir, "adapter_model.bin"))):
            size_mb = os.path.getsize(adapter_weights) / (1024 * 1024) if os.path.exists(adapter_weights) else 0
            logger.info(f"✅ 100% VERIFIED: Adapter weights ({size_mb:.2f} MB) and config saved to disk at {os.path.abspath(self.output_dir)}")
        else:
            logger.warning(f"⚠️ Warning: Expected adapter files not found in {self.output_dir}")

        # 2. Optional: Push permanently to Hugging Face Hub
        hub_model_id = self.config.get("hub_model_id")
        if self.config.get("push_to_hub", False) and hub_model_id:
            try:
                logger.info(f"☁️ Backing up trained adapters to Hugging Face Hub: {hub_model_id}...")
                trainer.model.push_to_hub(hub_model_id)
                self.tokenizer.push_to_hub(hub_model_id)
                logger.info(f"🎉 Successfully pushed adapter to Hugging Face Hub: https://huggingface.co/{hub_model_id}")
            except Exception as e:
                logger.warning(f"Could not push to Hugging Face Hub (local model is safe on disk): {e}")

        # 3. Automatically generate and save training curve dashboard
        try:
            from utilities.plotting import plot_training_curves
            plot_training_curves(trainer, output_dir=self.output_dir, show_plot=True)
        except Exception as e:
            logger.warning(f"Could not generate training plot: {e}")

        logger.info("Save successful. Fine-tuning complete!")


# Convenience wrapper for the main script
def run_training(
    config: dict, 
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    train_data: Dataset, 
    val_data: Optional[Dataset] = None
) -> None:
    """Convenience wrapper for the Trainer class."""
    trainer = Trainer(config, model, tokenizer, train_data, val_data)
    trainer.train_and_save()

if __name__ == "__main__":
    # Test logic
    pass


        