from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset, DatasetDict

import os
import logging
from typing import Tuple, Dict, Any

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaDataLoader:
    """
    A class responsible for loading, formatting, and splitting datasets
    especially for LLama-3 instruction tuning.
    """
    def __init__(self, config: Dict[str, Any], tokenizer: PreTrainedTokenizer):
        """
        Initializes the data loader.
        
        Args:
            config (dict): The configuration dictionary (usually loaded from a YAML file)
            tokenizer (PreTrainedTokenizer): The loaded LLaMA-3 tokenizer to apply chat templates.
        """
        self.config = config
        self.tokenizer = tokenizer

        # Extract config with defaults
        self.csv_path = self.config.get("dataset_path", None)
        self.split_ratio = self.config.get("test_size", 0.1)
        self.seed = self.config.get("seed", 42)

        # Limit the number of samples for fine-tuning due to GPU constraints
        self.max_samples = self.config.get("max_samples", 1000)

        self.text_column = "text"

    def _load_raw_data(self) -> Dataset:
        """
        Loads the csv file and subsamples it to fit GPU/time constraints.
        """
        if not self.csv_path or not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}. Please check the path in the configuration")

        logger.info(f"Loading dataset from {self.csv_path}")

        # The 'datasets' library handles large CSvs efficiently using memory mapping
        dataset = load_dataset("csv", data_files=self.csv_path, split="train")

        # Subsample the dataset to a manageable size
        if self.max_samples and self.max_samples < len(dataset):
            logger.info(f"Subsampling dataset from {len(dataset)} to {self.max_samples} samples for training")
            # Shuffle first to ensure a random distribution of data, then slice
            dataset = dataset.shuffle(seed=self.seed).select(range(self.max_samples))
        
        return dataset

    def _csv_to_messages(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts your specific 'input' and 'output' CSV columns into the 
        standard conversational messages format
        """
        system_msg = "You are a helpful and precise assistant."

        # Cast to string safely in case pandas/HF inferred empty cells as nulls/floats
        user_content = str(example.get("input", "")).strip()
        assistant_content = str(example.get("output", "")).strip()

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]

        return {"messages": messages}

    def _apply_llama3_template(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the LLaMA-3 chat template to the messages, 
        tokenizes the input, and formats the output for supervised fine-tuning.
        """
        # This returns the formatted string
        formatted_text = self.tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False, # Return string text; SFTTrainer does the tokenizing
            add_generation_prompt=False # We don't want the 'assistant:' tag added yet
        )
        return {self.text_column: formatted_text}

    def _format_dataset(self, dataset: Dataset) -> Dataset:
        """
        Pipelines the formatting methods across the entire dataset.
        """
        logger.info("Standardizing 'input'/'output' columns to conversational format..")
        # num_proc allows multi-processing to speed up mapping on large datasets
        dataset = dataset.map(
            self._csv_to_messages,
            desc='Converting to messages',
            num_proc = max(1, (os.cpu_count() or 1) - 2)
        )

        logger.info("Applying LLaMA-3 chat template...")
        dataset = dataset.map(
            self._apply_llama3_template,
            desc="Applying chat template",
            num_proc= max(1, (os.cpu_count() or 1) - 2)
        )

        # Drop the original columns to free up RAM before passing to the trainer
        dataset = dataset.select_columns([self.text_column])
        return dataset

    def _split_data(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """
        Splits the dataset into training and validation sets.
        """
        logger.info(f"Splitting dataset into train/validation (test_size={self.split_ratio})")
        dataset_dict = dataset.train_test_split(
            test_size=self.split_ratio,
            seed=self.seed
        )
        return dataset_dict["train"], dataset_dict["test"]

    def load_and_prepare(self) -> Tuple[Dataset, DatasetDict]:
        """
        Pipeline to load and prepare the dataset for training.
        """
        # Step 1: Load Raw Data
        dataset = self._load_raw_data()

        # Step 2: Format Dataset
        dataset = self._format_dataset(dataset)

        # Step 3: Split dataset
        train_data, val_data = self._split_data(dataset)

        logger.info(f"Data ready. Train size: {len(train_data)}, Val Size: {len(val_data)}")

        return train_data, val_data
