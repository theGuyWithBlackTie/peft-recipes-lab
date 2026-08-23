import os
import sys
import argparse
import yaml
import logging

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == "lora" else current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lora.src.model_builder import LlamaModelBuilder
from lora.src.data_loader import LlamaDataLoader
from lora.src.trainer import LlamaTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA on Hinglish using LoRA")
    parser.add_argument("--config", type=str, default=os.path.join(current_dir, "config.yaml"), help="Path to config YAML file")
    args = parser.parse_args()

    # 1. Load Configuration
    logger.info(f"Loading configuration from {args.config}...")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 2. Build LoRA Model and Tokenizer
    logger.info("Building LoRA Model...")
    model_builder = LlamaModelBuilder(config)
    model, tokenizer = model_builder.build()
    model.enable_input_require_grads()

    # 3. Load and Format Dataset
    logger.info("Loading and formatting dataset...")
    data_loader = LlamaDataLoader(config, tokenizer)
    train_data, val_data = data_loader.load_and_prepare()

    # 4. Train and Save
    logger.info("Starting training loop...")
    trainer = LlamaTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        val_dataset=val_data
    )
    trainer.train_and_save()
    logger.info("🎉 Fine-tuning finished successfully!")


if __name__ == "__main__":
    main()
