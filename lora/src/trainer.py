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
    using Hugging Face's SFTTrainer for LLama-3 QLoRA.
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
            model (PreTrainedModel): The PEFT-wrapped LLaMA model ready for QLoRA tuning
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

    def _get_training_arguments(self) -> TrainingArguments:
        """
        Maps configuration variables to Hugging Face TrainingArguments.
        Defaults are optimized for QLoRA fine-tuning on consumer GPUs.
        """
        logger.info("Configuring Training Arguments...")

        # evaluation strategy must match eval_steps logic
        do_eval = self.val_dataset is not None
        eval_strategy = "steps" if do_eval else "no"
        
        # evaluation steps logic
        eval_steps = self.config.get("eval_steps")
        if eval_steps is None:
            if do_eval:
                # default to 500 if not specified, but only if we have a validation set
                eval_steps = 500
            else:
                eval_steps = 0

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            # Batch size per GPU. Lower this if you get CUDA Out Of Memory (OOM) errors.
            per_device_train_batch_size=self.config.get("batch_size", 4),
            per_device_eval_batch_size=self.config.get("batch_size", 4),
            # Gradient accumulation simulates larger batch sizes by updating weights less frequently.
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            
            # Paged optimizers push optimizer states to CPU RAM when GPU VRAM runs out.
            optim="paged_adamw_8bit",
            learning_rate=float(self.config.get("learning_rate", 2e-4)),
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            max_grad_norm=0.3,
            
            num_train_epochs=self.config.get("epochs", 1),
            logging_steps=self.config.get("logging_steps", 10),
            
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=self.config.get("save_steps", 50),
            
            # LLaMA-3 natively uses bfloat16 (bf16). If your GPU is older than Ampere (RTX 30XX), 
            # set bf16=False and fp16=True instead.
            bf16=self.config.get("use_bf16", True),
            fp16=self.config.get("use_fp16", False)            
        )

    def _setup_trainer(self, training_args: TrainingArguments) -> SFTTrainer:
        """
        Instantiates the Supervised Fine-Tuning (SFT) Trainer
        """
        logging.info("Initializing SFTTrainer...")

        # Note: We pass peft_config=None because the model we pass in
        # is ALREADY wrapped with LoRA adapters from our model_builder.py script.
        return SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset, 
            dataset_text_field="text", # Maps to the column created in data_loader.py
            max_seq_length=self.config.get("max_seq_length", 2048),
            tokenizer=self.tokenizer,
            args=training_args,
            packing=False # Set to True to pack multiple short examples into one sequence for speed
        )

    def train_and_save(self) -> None:
        """
        Executes the training loop and saves the final adapters.
        """
        training_args = self._get_training_arguments()
        trainer = self._setup_trainer(training_args)

        logger.info("Starting training loop...")
        trainer.train()

        logger.info("Training complete. Saving final model adapters to {self.output_dir}...")
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
        