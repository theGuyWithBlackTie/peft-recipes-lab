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

    def _load_raw_data(self) -> Any:
        """
        Loads the dataset from Hugging Face Hub or a local file (CSV/Parquet).
        """
        dataset_path = self.csv_path or self.config.get("dataset_path", None)
        dataset_config = self.config.get("dataset_config", None)

        if not dataset_path:
            raise ValueError("No `dataset_path` specified in configuration.")

        logger.info(f"Loading dataset from: {dataset_path} (subset config: {dataset_config})...")

        # Check if loading from local file or Hugging Face Hub
        if os.path.exists(dataset_path):
            if dataset_path.endswith(".parquet"):
                dataset = load_dataset("parquet", data_files=dataset_path, split="train")
            else:
                dataset = load_dataset("csv", data_files=dataset_path, split="train")
        else:
            # Load directly from Hugging Face Hub
            if dataset_config:
                dataset = load_dataset(dataset_path, dataset_config)
            else:
                dataset = load_dataset(dataset_path)

        return dataset

    def _item_to_messages(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts 'input' and 'output' into standard conversational messages format.
        Supports both structured list-of-dicts format and plain text strings.
        """
        system_msg = self.config.get("system_prompt", "You are a helpful and friendly AI assistant that communicates in natural, conversational Hinglish.")

        inp = example.get("input")
        out = example.get("output")

        # Process input (structured list or string)
        if isinstance(inp, (list, tuple)) or (hasattr(inp, "__iter__") and not isinstance(inp, (str, bytes))):
            input_msgs = [dict(m) for m in inp]
        elif isinstance(inp, str):
            input_msgs = [{"role": "user", "content": inp.strip()}]
        else:
            input_msgs = []

        # Process output (structured list or string)
        if isinstance(out, (list, tuple)) or (hasattr(out, "__iter__") and not isinstance(out, (str, bytes))):
            output_msgs = [dict(m) for m in out]
        elif isinstance(out, str):
            output_msgs = [{"role": "assistant", "content": out.strip()}]
        else:
            output_msgs = []

        messages = [{"role": "system", "content": system_msg}] + input_msgs + output_msgs
        return {"messages": messages}

    def _apply_llama3_template(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the LLaMA-3 chat template to the messages, 
        tokenizes the input, and formats the output for supervised fine-tuning.
        """
        formatted_text = self.tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {self.text_column: formatted_text}

    def _format_dataset(self, dataset: Dataset) -> Dataset:
        """
        Pipelines the formatting methods across the entire dataset.
        """
        logger.info("Standardizing 'input'/'output' columns to conversational format...")
        dataset = dataset.map(
            self._item_to_messages,
            desc="Converting to messages",
            num_proc=max(1, (os.cpu_count() or 1) - 2)
        )

        logger.info("Applying LLaMA-3 chat template...")
        dataset = dataset.map(
            self._apply_llama3_template,
            desc="Applying chat template",
            num_proc=max(1, (os.cpu_count() or 1) - 2)
        )

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

    def load_and_prepare(self) -> Tuple[Dataset, Dataset]:
        """
        Pipeline to load and prepare the dataset for training.
        """
        raw_data = self._load_raw_data()

        # Case 1: Dataset loaded as a DatasetDict with pre-existing train/test splits (e.g. from HF Hub)
        if isinstance(raw_data, DatasetDict):
            train_key = "train" if "train" in raw_data else list(raw_data.keys())[0]
            val_key = "test" if "test" in raw_data else ("validation" if "validation" in raw_data else list(raw_data.keys())[-1])
            
            train_dataset = raw_data[train_key]
            val_dataset = raw_data[val_key]

            if self.max_samples and self.max_samples < len(train_dataset):
                logger.info(f"Subsampling train dataset from {len(train_dataset)} to {self.max_samples}")
                train_dataset = train_dataset.shuffle(seed=self.seed).select(range(self.max_samples))

            train_data = self._format_dataset(train_dataset)
            val_data = self._format_dataset(val_dataset)
        else:
            # Case 2: Single Dataset that requires in-memory splitting
            if self.max_samples and self.max_samples < len(raw_data):
                logger.info(f"Subsampling dataset from {len(raw_data)} to {self.max_samples}")
                raw_data = raw_data.shuffle(seed=self.seed).select(range(self.max_samples))

            formatted_dataset = self._format_dataset(raw_data)
            train_data, val_data = self._split_data(formatted_dataset)

        logger.info(f"Data ready. Train size: {len(train_data)}, Val Size: {len(val_data)}")
        return train_data, val_data
