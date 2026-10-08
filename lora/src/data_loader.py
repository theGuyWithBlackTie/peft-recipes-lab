from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import os
import logging
from typing import Tuple, Dict, Any

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    A class responsible for loading and formatting datasets for Medical CoT fine-tuning.
    """
    def __init__(self, config: Dict[str, Any], tokenizer: PreTrainedTokenizer):
        """
        Initializes the data loader.
        
        Args:
            config (dict): The configuration dictionary.
            tokenizer (PreTrainedTokenizer): The loaded model tokenizer.
        """
        self.config = config
        self.tokenizer = tokenizer

        self.dataset_path = self.config.get("dataset_path", "theguywithblacktie/medical-cot")
        self.local_fallback = self.config.get("local_dataset_path_fallback", "dataset/hf_export/medical_cot")
        self.experiment_mode = self.config.get("experiment_mode", "cot")
        self.split_ratio = self.config.get("test_size", 0.1)
        self.seed = self.config.get("seed", 42)
        self.max_samples = self.config.get("max_samples", None)

        if self.experiment_mode == "direct":
            self.source_text_column = "text_direct"
        else:
            self.source_text_column = "text_cot"

    def _format_cot_fallback(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formatting fallback if loading from ty-kim/medical_cot.
        """
        instruction = example.get('question', example.get('instruction', ''))
        complex_cot = example.get('Complex_CoT', '')
        response = example.get('Response', '')

        text_cot = f"[INST] {instruction} [/INST] <think>\n{complex_cot}\n</think>\n{response}"
        text_direct = f"[INST] {instruction} [/INST] {response}"

        return {
            "text_cot": text_cot,
            "text_direct": text_direct
        }

    def _load_raw_data(self) -> Any:
        """
        Loads the dataset from Hugging Face Hub, local export, or falls back to raw ty-kim/medical_cot.
        """
        try:
            logger.info(f"Attempting to load dataset from HF Hub: {self.dataset_path}")
            dataset = load_dataset(self.dataset_path)
            return dataset
        except Exception as e:
            logger.warning(f"Failed to load from HF Hub: {e}. Trying local fallback: {self.local_fallback}")
            if os.path.exists(self.local_fallback):
                try:
                    dataset = load_from_disk(self.local_fallback)
                    return dataset
                except Exception as e2:
                    logger.warning(f"Failed to load from local fallback: {e2}. Trying original ty-kim/medical_cot...")
            else:
                logger.warning(f"Local fallback path {self.local_fallback} does not exist. Trying original ty-kim/medical_cot...")
            
            dataset = load_dataset("ty-kim/medical_cot", split="train")
            dataset = dataset.map(self._format_cot_fallback)
            return dataset.train_test_split(test_size=self.split_ratio, seed=self.seed)

    def load_and_prepare(self) -> Tuple[Dataset, Dataset]:
        """
        Pipeline to load and prepare the dataset for training.
        """
        raw_data = self._load_raw_data()

        if isinstance(raw_data, DatasetDict):
            train_key = "train" if "train" in raw_data else list(raw_data.keys())[0]
            val_key = "test" if "test" in raw_data else ("validation" if "validation" in raw_data else list(raw_data.keys())[-1])
            
            train_dataset = raw_data[train_key]
            val_dataset = raw_data[val_key]
        else:
            # If a single split is returned
            split = raw_data.train_test_split(test_size=self.split_ratio, seed=self.seed)
            train_dataset = split["train"]
            val_dataset = split["test"]

        # Rename the selected column to 'text' to standardize for SFTTrainer
        def rename_col(example):
            return {"text": example[self.source_text_column]}

        train_dataset = train_dataset.map(rename_col, remove_columns=train_dataset.column_names)
        val_dataset = val_dataset.map(rename_col, remove_columns=val_dataset.column_names)

        if self.max_samples and self.max_samples < len(train_dataset):
            logger.info(f"Subsampling train dataset from {len(train_dataset)} to {self.max_samples}")
            train_dataset = train_dataset.shuffle(seed=self.seed).select(range(self.max_samples))

        max_eval = self.config.get("max_eval_samples", 200)
        if max_eval and val_dataset is not None and len(val_dataset) > max_eval:
            logger.info(f"Subsampling validation dataset from {len(val_dataset)} to {max_eval} for fast evaluation...")
            val_dataset = val_dataset.shuffle(seed=self.seed).select(range(max_eval))

        logger.info(f"Data ready. Train size: {len(train_dataset)}, Val Size: {len(val_dataset)}")
        return train_dataset, val_dataset
