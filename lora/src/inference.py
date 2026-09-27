from typing import Dict, Any, List, Optional
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    A class for running inference using a base model combined with
    fine-tuned LoRA adapters.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the inference wrapper

        Args:
            config (dict): Project configuration dictionary
        """
        self.config = config
        self.model_id = self.config.get("model_id", "mistralai/Mistral-7B-Instruct-v0.3")
        self.adapter_path = self.config.get("output_dir", "./mistral-lora-outputs")

        self.tokenizer = None
        self.model = None

    def _load_model_and_tokenizer(self) -> None:
        """Loads the base model and merges/attaches the LoRA adapters."""
        logger.info(f"Loading tokenizer from {self.adapter_path} (fallback to {self.model_id})...")

        # Try loading tokenizer from the adapter path first (saved during training)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path, use_fast=True)
            logger.info("Tokenizer loaded from adapter path.")
        except Exception as e:
            logger.warning(f"Could not load tokenizer from adapter path: {e}")
            logger.info(f"Loading tokenizer from base model: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        native_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
        requested_bf16 = self.config.get("use_bf16", False) and native_bf16
        requested_fp16 = self.config.get("use_fp16", True)

        torch_dtype = (
            torch.bfloat16 if requested_bf16
            else (torch.float16 if (requested_fp16 and torch.cuda.is_available()) else torch.float32)
        )

        use_4bit = self.config.get("use_4bit", True)
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        logger.info(f"Loading base model {self.model_id} with torch_dtype={torch_dtype}, use_4bit={use_4bit}...")
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": self.config.get("device_map", "auto"),
            "trust_remote_code": True
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )

        logger.info(f"Loading and applying LoRA adapters from {self.adapter_path}...")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)

        # Set the model to evaluation mode for inference
        self.model.eval()
        logger.info("Model and tokenizer successfully loaded for inference.")

    def generate(self, user_prompt: str = None, messages: List[Dict[str, str]] = None, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Generate a response using the model's chat template.
        Pass either `user_prompt` for a single turn, or `messages` for a full conversation history.
        """
        if self.model is None or self.tokenizer is None:
            self._load_model_and_tokenizer()
        
        # Construct the message payload if not provided
        if messages is None:
            if user_prompt is None:
                raise ValueError("Either `user_prompt` or `messages` must be provided.")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        logger.info("Tokenizing input...")
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        logger.info("Generating response...")
        # Collect valid terminators dynamically across architectures
        candidate_terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("</s>"),
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        ]
        terminators = [t for t in candidate_terminators if t is not None and isinstance(t, int)]

        # Use torch.no_grad() to save memory since we aren't calculating gradients
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=int(self.config.get("max_new_tokens", 256)),
                temperature=float(self.config.get("temperature", 0.6)),
                top_p=float(self.config.get("top_p", 0.9)),
                repetition_penalty=float(self.config.get("repetition_penalty", 1.15)),
                do_sample=True,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Slice the output tokens to ignore the prompt tokens and only decode the new generated text
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    def chat_loop(self) -> None:
        """Starts a simple terminal-based chat loop for continuous testing."""
        if self.model is None or self.tokenizer is None:
            self._load_model_and_tokenizer()
        
        print("\n--- Starting Chat Session (type 'quit' or 'exit' to stop) ---")
        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting chat.")
                break
            
            messages.append({"role": "user", "content": user_input})
            response = self.generate(messages=messages)
            print(f"\nModel: {response}")
            messages.append({"role": "assistant", "content": response})

# Convenience wrapper for testing
def run_chat(config: dict) -> None:
    """Convenience wrapper to initialize and run the chat loop."""
    inference_engine = InferenceEngine(config)
    inference_engine.chat_loop()

if __name__ == "__main__":
    sample_config = {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "output_dir": "./mistral-lora-outputs",
        "max_new_tokens": 256,
        "temperature": 0.6
    }
    run_chat(sample_config)


        