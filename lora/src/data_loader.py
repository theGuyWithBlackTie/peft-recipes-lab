from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset, DatasetDict

import os
import logging
from typing import Tuple, Dict, Any

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    A class responsible for loading, formatting, and splitting datasets
    for instruction and conversational fine-tuning.
    """
    def __init__(self, config: Dict[str, Any], tokenizer: PreTrainedTokenizer):
        """
        Initializes the data loader.
        
        Args:
            config (dict): The configuration dictionary (usually loaded from a YAML file)
            tokenizer (PreTrainedTokenizer): The loaded model tokenizer to apply chat templates.
        """
        self.config = config
        self.tokenizer = tokenizer

        # Extract config with defaults
        self.csv_path = self.config.get("dataset_path", None)
        self.split_ratio = self.config.get("test_size", 0.1)
        self.seed = self.config.get("seed", 42)

        # Limit the number of samples for fine-tuning due to GPU constraints
        self.max_samples = self.config.get("max_samples", None)

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
        Strictly normalizes roles and enforces alternating user/assistant turns
        to comply with Mistral and other strict chat templates.
        """
        system_msg = self.config.get("system_prompt", "You are a helpful and friendly AI assistant that communicates in natural, conversational Hinglish.")

        inp = example.get("input")
        out = example.get("output")

        raw_turns = []
        # 1. Extract input turns
        if isinstance(inp, (list, tuple)) or (hasattr(inp, "__iter__") and not isinstance(inp, (str, bytes))):
            for m in inp:
                if isinstance(m, dict):
                    role = str(m.get("role", "")).strip().lower()
                    content = str(m.get("content", "")).strip()
                    if content:
                        raw_turns.append((role, content))
        elif isinstance(inp, str) and inp.strip():
            raw_turns.append(("user", inp.strip()))

        # 2. Extract output turns
        if isinstance(out, (list, tuple)) or (hasattr(out, "__iter__") and not isinstance(out, (str, bytes))):
            for m in out:
                if isinstance(m, dict):
                    role = str(m.get("role", "")).strip().lower()
                    content = str(m.get("content", "")).strip()
                    if content:
                        raw_turns.append((role, content))
        elif isinstance(out, str) and out.strip():
            raw_turns.append(("assistant", out.strip()))

        # 3. Normalize roles (user/human -> user, assistant/gpt/bot/model -> assistant)
        # and merge consecutive turns with identical roles
        normalized = []
        for role, content in raw_turns:
            if role in ["user", "human"]:
                norm_role = "user"
            elif role in ["assistant", "gpt", "bot", "model"]:
                norm_role = "assistant"
            elif role == "system":
                system_msg = f"{system_msg} {content}".strip()
                continue
            else:
                norm_role = "assistant" if (normalized and normalized[-1]["role"] == "user") else "user"

            if normalized and normalized[-1]["role"] == norm_role:
                normalized[-1]["content"] = f"{normalized[-1]['content']}\n\n{content}"
            else:
                normalized.append({"role": norm_role, "content": content})

        # 4. Strict Alternation Enforcement:
        # Mistral requirement: After optional system, roles MUST alternate: user -> assistant -> user -> assistant
        convo = []
        expected_role = "user"
        for turn in normalized:
            if turn["role"] == expected_role:
                convo.append(turn)
                expected_role = "assistant" if expected_role == "user" else "user"
            elif convo and turn["role"] == convo[-1]["role"]:
                convo[-1]["content"] = f"{convo[-1]['content']}\n\n{turn['content']}"

        # 5. Guarantee at least 1 valid user turn and 1 valid assistant turn
        if not convo or convo[0]["role"] != "user":
            fallback_u = inp if isinstance(inp, str) and inp.strip() else "Hello"
            convo.insert(0, {"role": "user", "content": fallback_u})

        if convo[-1]["role"] != "assistant":
            fallback_a = out if isinstance(out, str) and out.strip() else "Haan, bataiye main aapki kya madad kar sakta hoon?"
            convo.append({"role": "assistant", "content": fallback_a})

        # 6. Final verification of alternation
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg.strip()})
        messages.extend(convo)

        return {"messages": messages}

    def _apply_chat_template(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the tokenizer's chat template to the messages, 
        tokenizes the input, and formats the output for supervised fine-tuning.
        """
        try:
            formatted_text = self.tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            # Fallback for any template parsing edge cases: strip system and wrap in clean user/assistant
            msgs = example["messages"]
            user_parts = [m["content"] for m in msgs if m["role"] in ["system", "user"]]
            asst_parts = [m["content"] for m in msgs if m["role"] == "assistant"]
            clean_msgs = [
                {"role": "user", "content": "\n\n".join(user_parts)},
                {"role": "assistant", "content": "\n\n".join(asst_parts)}
            ]
            formatted_text = self.tokenizer.apply_chat_template(
                clean_msgs,
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

        logger.info("Applying model chat template...")
        dataset = dataset.map(
            self._apply_chat_template,
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

        max_eval = self.config.get("max_eval_samples", 200)
        if max_eval and val_data is not None and len(val_data) > max_eval:
            logger.info(f"Subsampling validation dataset from {len(val_data)} to {max_eval} for fast evaluation...")
            val_data = val_data.shuffle(seed=self.seed).select(range(max_eval))

        logger.info(f"Data ready. Train size: {len(train_data)}, Val Size: {len(val_data)}")
        return train_data, val_data


