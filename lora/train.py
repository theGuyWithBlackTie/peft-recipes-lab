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

from lora.src.model_builder import ModelBuilder
from lora.src.data_loader import DataLoader
from lora.src.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def authenticate_hf():
    """Ensures child processes in DDP are authenticated with Hugging Face."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HF_TOKEN")
        except Exception:
            pass
    if not hf_token:
        try:
            from google.colab import userdata
            hf_token = userdata.get("HF_TOKEN")
        except Exception:
            pass
    if hf_token:
        from huggingface_hub import login
        os.environ["HF_TOKEN"] = hf_token
        login(token=hf_token)
        logger.info("✅ Authenticated with Hugging Face Hub.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on Hinglish using LoRA / QLoRA")
    parser.add_argument("--config", type=str, default=os.path.join(current_dir, "config.yaml"), help="Path to config YAML file")
    args = parser.parse_args()

    # 1. Authenticate with Hugging Face
    authenticate_hf()

    # 2. Load Configuration
    logger.info(f"Loading configuration from {args.config}...")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 3. Build LoRA Model and Tokenizer
    logger.info("Building LoRA Model...")
    model_builder = ModelBuilder(config)
    model, tokenizer = model_builder.build()
    model.enable_input_require_grads()

    # 4. Load and Format Dataset
    logger.info("Loading and formatting dataset...")
    data_loader = DataLoader(config, tokenizer)
    train_data, val_data = data_loader.load_and_prepare()

    # 5. Train and Save
    logger.info("Starting training loop...")
    trainer = Trainer(
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

