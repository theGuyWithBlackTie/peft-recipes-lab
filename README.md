# 🧪 PEFT Recipes Lab

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face PEFT](https://img.shields.io/badge/%F0%9F%A4%97%20PEFT-0.10%2B-yellow)](https://github.com/huggingface/peft)
[![TRL SFTTrainer](https://img.shields.io/badge/TRL-SFTTrainer-orange)](https://github.com/huggingface/trl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hardware: NVIDIA T4 / A100 / H100](https://img.shields.io/badge/Hardware-NVIDIA%20T4%20%7C%20A100-76B900.svg?logo=nvidia&logoColor=white)](https://www.nvidia.com/)

**Production-grade, modular, and reproducible Parameter-Efficient Fine-Tuning (PEFT) recipes for modern Large Language Models.**

[Overview](#-overview--mission) • [Recipe 1: LoRA](#-recipe-1-lora-low-rank-adaptation) • [Repository Structure](#-repository-structure) • [Quickstart](#-quickstart) • [Evaluation Suite](#-comprehensive-evaluation-suite) • [Speed Optimization](#-speed-optimization--hardware-tuning) • [PEFT Roadmap](#-peft-recipes-roadmap)

</div>

---

## 🎯 Overview & Mission

**`PEFT Recipes Lab`** is an open-source engineering lab dedicated to practical, high-throughput Parameter-Efficient Fine-Tuning recipes for modern open-weights LLMs.

Fine-tuning LLMs on custom domains, dialects, and tasks often suffers from slow training iterations, excessive GPU memory consumption, and unstructured ad-hoc evaluation. This repository provides:

1. **Modular Architecture:** Clean separation of data loading, model building, multi-precision training, and inference across different PEFT techniques.
2. **10x Speed Optimization Engine:** Eliminates hardware emulation bottlenecks on commodity GPUs (e.g. NVIDIA T4, V100, RTX 4090) through Scaled Dot-Product Attention (`sdpa`), dynamic hardware reflection, and memory optimization.
3. **Rigorous Multi-Pillar Evaluation Suite:** Quantitative Perplexity ($PPL = e^{\text{Loss}}$), character-level script purity, and multi-dimensional LLM-as-a-Judge with blind head-to-head win-rate benchmarking.
4. **Automated Artifacts & Dashboards:** Zero-overhead generation of publication-ready training loss/perplexity curves and structured JSON metrics.

---

## 🌟 Recipe 1: LoRA (Low-Rank Adaptation)

Our first completed recipe adapts **Meta's `Llama-3.2-1B-Instruct`** to converse naturally in **Romanized Hinglish** (Hindi code-mixed with English in clean Latin script) using conversational dialogues from [`theguywithblacktie/hinglish-conversations`](https://huggingface.co/datasets/theguywithblacktie/hinglish-conversations).

### 💬 Before vs. After LoRA Comparison

| User Query | Base Model (`Llama-3.2-1B-Instruct`) | LoRA Fine-Tuned (`PEFT Recipes Lab`) |
|---|---|---|
| *"arre yaar, aaj office mein itna zyada kaam tha ki dimag ka dahi ho gaya. thoda relax karne ka koi mast idea batao."* | Formal English advice or literal, awkward textbook Hindi translations. | *"haan yaar, try kar lo ek chhoti si walk ya apne favorite snacks ke saath mast movie dekh lo. aaj raat kaam bilkul mat sochna, dimag ko thoda chill time do!"* |
| *"weekend pe doston ke saath kuch accha cook karna hai. koi easy aur spicy street food style recipe suggest karo na!"* | Generic English recipes without cultural context. | *"try kar lo 'chickpea tikka masala' ya quick street-style papdi chaat! super easy aur mast spicy banta hai, doston ko bohot pasand aayega!"* |

---

## ✨ Key Features

* **⚡ Dynamic Hardware Precision Detection:** Automatically detects whether the host GPU has native hardware support for `bfloat16` (`torch.cuda.is_bf16_supported(including_emulation=False)`). Prevents 10x slow software emulation on Turing GPUs (T4) by routing to native FP16 Tensor Cores, while taking full advantage of BF16 on Ampere/Hopper (A100/H100/L4).
* **🧠 LLaMA-3 Stop-Token Compliance:** Correctly handles LLaMA-3 chat terminators (`<|end_of_text|>` and `<|eot_id|>`), preventing autoregressive repetitive loops and token degeneration.
* **📊 4-Panel Training Visualizer:** Automatically logs and plots Step Loss, Step-level Perplexity, Gradient Norm stability, and Learning Rate schedules to `training_curves.png` and `training_metrics.json`.
* **💾 Guaranteed Disk Verification & Hub Auto-Sync:** Verifies local `.safetensors` weight serializations before pushing cleanly to Hugging Face Model Hub.
* **🚀 Multi-GPU & Single-GPU Compatible:** Seamlessly runs in interactive notebooks (Colab/Kaggle) with `device_map="auto"` or across multi-GPU clusters via `accelerate launch`.

---

## 📂 Repository Structure

```tree
peft-recipes-lab/
├── lora/                                # Recipe 1: LoRA Fine-Tuning Implementation
│   ├── config.yaml                      # Hyperparameters, precision, and training configuration
│   ├── train.py                         # Multi-GPU CLI runner with DDP & accelerate support
│   ├── finetune-mistral-7b.ipynb        # Interactive end-to-end training & evaluation notebook
│   └── src/
│       ├── data_loader.py               # Chat template formatter & deterministic subsampler
│       ├── model_builder.py             # QLoRA 4-bit / precision selector & adapter injector
│       ├── trainer.py                   # SFTTrainer wrapper, integrity verifier & curve logger
│       └── inference.py                 # Chat inference engine with architecture-safe stop tokens
│
├── evaluation/                          # Unified 3-Pillar Evaluation Suite (Shared across all recipes)
│   ├── perplexity.py                    # Cross-Entropy Loss & Perplexity (PPL = exp(loss))
│   ├── script_purity.py                 # Latin vs Devanagari character-level adherence detector
│   ├── judge_prompts.py                 # Prompt templates for single-turn & pairwise LLM-as-a-Judge
│   ├── llm_judge.py                     # Multi-provider judge engine (OpenAI, Gemini, Anthropic)
│   └── evaluator.py                     # Consolidated markdown evaluation report generator
│
├── utilities/                           # Common Shared Utilities
│   └── plotting.py                      # 4-Panel dashboard visualizer & JSON metrics exporter
│
├── Pipfile                              # Pinned dependencies & version bounds
└── README.md                            # Repository documentation
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/theGuyWithBlackTie/peft-recipes-lab.git
cd peft-recipes-lab

# Using Pipenv
pipenv install --dev

# Or using pip
pip install -r <(pipenv requirements)
```

---

### 2. Interactive Notebook Execution (Kaggle / Google Colab)

To run the LoRA recipe directly inside a cloud notebook:

```python
# 1. Clone the repository directly inside the notebook
!git clone https://github.com/theGuyWithBlackTie/peft-recipes-lab.git
import os, sys
sys.path.insert(0, '/kaggle/working/peft-recipes-lab')
os.chdir('/kaggle/working/peft-recipes-lab')

# 2. Open and run lora/finetune-mistral-7b.ipynb
```

---

### 3. Multi-GPU Distributed Training via CLI

Launch distributed training across multiple GPUs using Hugging Face `accelerate`:

```bash
# Set your Hugging Face Access Token
export HF_TOKEN="hf_your_access_token"

# Launch multi-GPU distributed SFT
accelerate launch lora/train.py --config lora/config.yaml
```

---

### 4. Interactive Inference & Chat

Run inference using the fine-tuned adapter:

```python
from lora.src.inference import InferenceEngine

config = {
    "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
    "output_dir": "./mistral-lora-outputs",
    "use_4bit": True,
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "system_prompt": "You are a helpful assistant that converses naturally in Romanized Hinglish."
}

engine = InferenceEngine(config)

response = engine.generate(
    user_prompt="bhai kal exam hai aur abhi tak kuch nahi padha, kya karun?"
)
print("🤖 Assistant:", response)
```

---

## 📊 Comprehensive Evaluation Suite

The `evaluation/` module provides a comprehensive 3-pillar benchmarking framework shared across all PEFT recipes:

```python
from evaluation.evaluator import HinglishEvaluator

evaluator = HinglishEvaluator(judge_model_name="gpt-4o-mini")

test_samples = [
    {
        "prompt": "weekend pe doston ke saath kuch accha cook karna hai.",
        "response_base": "Here is a recipe for spaghetti...",
        "response_lora": "haan try kar lo chickpea tikka masala ya papdi chaat!"
    }
]

report = evaluator.evaluate_generation_quality(test_samples, run_llm_judge=True)
print(evaluator.generate_markdown_summary(report))
```

### 🏆 Benchmark Metrics:
* **Perplexity Reduction:** Quantifies confidence gain on domain language ($PPL_{\text{reduction}} = \frac{PPL_{\text{base}} - PPL_{\text{lora}}}{PPL_{\text{base}}} \times 100$).
* **Script Purity Score:** Measures percentage of Latin characters vs accidental Devanagari character bleed.
* **LLM-as-a-Judge:** 
  * *Coherence & Sensicality (1–5)*
  * *Helpfulness & Factuality (1–5)*
  * *Hinglish Naturalness & Flow (1–5)*
  * *Blind Head-to-Head Win Rate (%)*

---

## ⚡ Speed Optimization & Hardware Tuning

| Parameter / Lever | 🐢 Default Setup (~3 Hours) | ⚡ Optimized Setup (~20 Mins) | Technical Reason |
|---|---|---|---|
| **Precision on T4** | `use_bf16: true` | `use_fp16: true, use_bf16: false` | Turing lacks native BF16; emulated BF16 is 10x slower than FP16 Tensor Cores. |
| **Attention Implementation** | Eager Attention | `attn_implementation="sdpa"` | Scaled Dot-Product Attention fuses CUDA kernels, avoiding intermediate matrix allocation. |
| **Sequence Length ($L$)** | `max_seq_length: 2048` | `max_seq_length: 512` (or `1024`) | Attention FLOPs scale as $O(L^2)$. Conversational turns average ~250 tokens. |
| **Gradient Checkpointing** | `true` | `false` (for 1B–3B models) | Eliminates the ~30% compute overhead of activation recomputation during backprop. |
| **Per-Device Batch Size** | `batch_size: 4` | `batch_size: 4, grad_accum: 4` | Maximizes Tensor Core saturation while fitting within 16GB VRAM. |

---

## 🗺️ PEFT Recipes Roadmap

This repository is actively developing dedicated recipe implementations for each major PEFT paradigm:

- [x] **Recipe 1: LoRA (Low-Rank Adaptation):** Standard low-rank weight decomposition on LLaMA-3.2-1B.
- [ ] **Recipe 2: QLoRA (Quantized LoRA):** 4-bit NormalFloat (NF4) base model quantization with double quantization.
- [ ] **Recipe 3: AdaLoRA (Adaptive Low-Rank Adaptation):** Dynamic rank allocation across weight matrices based on importance scores.
- [ ] **Recipe 4: Prefix Tuning & Prompt Tuning:** Prepending trainable continuous virtual tokens.
- [ ] **Recipe 5: DoRA (Weight-Decomposed Low-Rank Adaptation):** Decomposing weights into magnitude and directional updates.
- [ ] **Recipe 6: Alignment via DPO / ORPO:** Direct Preference Optimization for Hinglish alignment pairs.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ for open-source AI and multilingual NLP research.
</div>
