# 🧪 PEFT Recipes Lab

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face PEFT](https://img.shields.io/badge/%F0%9F%A4%97%20PEFT-0.10%2B-yellow)](https://github.com/huggingface/peft)
[![TRL SFTTrainer](https://img.shields.io/badge/TRL-SFTTrainer-orange)](https://github.com/huggingface/trl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hardware: NVIDIA P100 / T4 / A100](https://img.shields.io/badge/Hardware-NVIDIA%20P100%20%7C%20T4%20%7C%20A100-76B900.svg?logo=nvidia&logoColor=white)](https://www.nvidia.com/)

**Production-grade, modular, and reproducible Parameter-Efficient Fine-Tuning (PEFT) recipes for modern Large Language Models.**

[Overview](#-overview--mission) • [Recipes](#-completed-recipes) • [Repository Structure](#-repository-structure) • [Quickstart](#-quickstart) • [Evaluation Suite](#-comprehensive-evaluation-suite) • [Speed Optimization](#-speed-optimization--hardware-tuning) • [Roadmap](#-peft-recipes-roadmap)

</div>

---

## 🎯 Overview & Mission

**`PEFT Recipes Lab`** is an open-source engineering lab dedicated to practical, high-throughput Parameter-Efficient Fine-Tuning recipes for modern open-weights LLMs.

Fine-tuning LLMs on custom domains, dialects, and tasks often suffers from slow training iterations, excessive GPU memory consumption, and unstructured ad-hoc evaluation. This repository provides:

1. **Modular Architecture:** Clean separation of data loading, model building, multi-precision training, and inference across different PEFT techniques.
2. **Speed & Memory Optimization Engine:** Eliminates hardware bottlenecks on commodity GPUs (e.g. NVIDIA P100, T4, RTX 4090) through Scaled Dot-Product Attention (`sdpa`), 4-bit NF4 quantization, and left-padded batched generation with KV-caching.
3. **Rigorous Multi-Pillar Evaluation Suite:** Quantitative Perplexity ($PPL = e^{\text{Loss}}$), character-level script purity, and multi-dimensional LLM-as-a-Judge with blind head-to-head win-rate benchmarking.
4. **Automated Artifacts & Dashboards:** Zero-overhead generation of publication-ready training loss/perplexity curves and structured JSON metrics.

---

## 🌟 Completed Recipes

### 1. Recipe 1: LoRA (Low-Rank Adaptation) — `Llama-3.2-1B-Instruct`
Standard full-precision / FP16 low-rank weight decomposition adapting Meta's `Llama-3.2-1B-Instruct` on Romanized Hinglish dialogues from [`theguywithblacktie/hinglish-conversations`](https://huggingface.co/datasets/theguywithblacktie/hinglish-conversations).

### 2. Recipe 2: 4-bit QLoRA (Quantized LoRA) — `Mistral-7B-Instruct-v0.3`
High-efficiency 4-bit NormalFloat (NF4) quantization with double quantization on Mistral-7B. Fine-tuned on 64,000+ conversational samples on a single 16GB GPU (NVIDIA P100 / T4) while fitting comfortably under 6GB VRAM.

#### 💬 Before vs. After Fine-Tuning Comparison (Mistral-7B)

| User Query | Base Mistral-7B (Before) | QLoRA Fine-Tuned (After) |
|---|---|---|
| *"arre yaar, aaj office mein itna zyada kaam tha ki dimag ka dahi ho gaya. thoda relax karne ka koi mast idea batao."* | Formal English advice or literal, awkward textbook Hindi translations. | *"haan yaar, tension mat lo! ek garam cup chai ya coffee piyo, thodi der phone side rakh do, aur apna favorite comedy show ya music chala lo. dimag ekdum fresh ho jayega!"* |
| *"weekend pe doston ke saath kuch accha cook karna hai. koi easy aur spicy street food style recipe suggest karo na!"* | Generic Western recipe suggestions without cultural context. | *"try karo street-style Pav Bhaji ya quick Crispy Corn Chaat! bohot jaldi banta hai, spicy aur chatpata hota hai, doston ko super pasand aayega!"* |
| *"bhai ek confusion hai, kya mujhe apna personal laptop upgrade karna chahiye ya cloud setup use karna chahiye data science ke liye?"* | Generic computer hardware breakdown in pure English. | *"agar basic learning aur tabular data hai toh laptop theek hai, par LLMs ya heavy deep learning ke liye local GPU pe kharcha karne se accha Colab ya Kaggle cloud setup use karo—paisa aur time dono bachega!"* |

---

## ✨ Key Features

* **⚡ Dynamic Hardware Precision Detection:** Automatically detects whether the host GPU has native hardware support for `bfloat16` (`torch.cuda.is_bf16_supported(including_emulation=False)`). Prevents 10x slow software emulation on Turing/Pascal GPUs by routing to native FP16 Tensor Cores, while taking full advantage of BF16 on Ampere/Hopper (A100/H100/L4).
* **🧠 Architecture-Safe Chat Formatting:** Enforces strict role alternation (`user -> assistant -> user`) required by Mistral and LLaMA Jinja chat templates, automatically normalizing roles and merging consecutive messages.
* **🚀 15x Fast Batched Inference:** Left-padded parallel batch generation (`batch_size=8`) with KV-caching (`use_cache=True`) and explicit EOS terminators, cutting inference time from 50s down to ~3s per pair.
* **📊 4-Panel Training Visualizer:** Automatically logs and plots Step Loss, Step-level Perplexity, Gradient Norm stability, and Learning Rate schedules to `training_curves.png` and `training_metrics.json`.
* **💾 Guaranteed Integrity & Hub Sync:** Verifies local `.safetensors` weight serializations before pushing cleanly to Hugging Face Model Hub.

---

## 📂 Repository Structure

```tree
peft-recipes-lab/
├── lora/                                # LoRA & QLoRA Fine-Tuning Implementation
│   ├── config.yaml                      # Hyperparameters, precision, and training configuration
│   ├── train.py                         # Multi-GPU CLI runner with DDP & accelerate support
│   ├── finetune-mistral-7b.ipynb        # Mistral-7B QLoRA training notebook
│   ├── evaluate-mistral-7b.ipynb        # Fast batched inference & perplexity evaluation notebook
│   ├── finetune-llama_3.2-1b.ipynb      # LLaMA-3.2-1B LoRA training notebook
│   └── src/
│       ├── data_loader.py               # Conversational dataset formatter & deterministic subsampler
│       ├── model_builder.py             # QLoRA 4-bit / precision selector & adapter injector
│       ├── trainer.py                   # SFTTrainer wrapper, integrity verifier & curve logger
│       └── inference.py                 # Chat inference engine with architecture-safe stop tokens
│
├── evaluation/                          # Unified 3-Pillar Evaluation Suite (Shared across all recipes)
│   ├── perplexity.py                    # Cross-Entropy Loss & Perplexity (PPL = exp(loss)) with adapter toggling
│   ├── script_purity.py                 # Latin vs Devanagari character-level adherence detector
│   ├── judge_prompts.py                 # Prompt rubrics for single-turn & pairwise LLM-as-a-Judge
│   ├── llm_judge.py                     # Multi-provider judge engine (Google Gemini, OpenAI, Claude) with retry backoff
│   └── evaluator.py                     # Consolidated HinglishEvaluator dashboard generator
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

# Or using standard pip
pip install -r <(pipenv requirements)
```

---

### 2. Interactive Notebook Training (Kaggle / Google Colab)

To fine-tune Mistral-7B using 4-bit QLoRA on a single 16GB GPU:

```python
# 1. Clone repository directly inside Kaggle / Colab
!git clone https://github.com/theGuyWithBlackTie/peft-recipes-lab.git
import sys
sys.path.insert(0, '/kaggle/working/peft-recipes-lab')

# 2. Open and run lora/finetune-mistral-7b.ipynb
```

---

### 3. Evaluation & Inference (Kaggle + Local)

Run [`lora/evaluate-mistral-7b.ipynb`](lora/evaluate-mistral-7b.ipynb) to evaluate the fine-tuned model:

1. **Quantitative Perplexity:** Computes cross-entropy loss and perplexity on test data for Base Model vs. LoRA Model using zero-overhead in-memory adapter toggling (`model.disable_adapter()`).
2. **Batched Response Generation:** Generates side-by-side responses on 100 test prompts using parallel left-padded decoding (`batch_size=8`) in ~10 minutes.
3. **Run Evaluation:** Pass the responses directly into [`HinglishEvaluator`](evaluation/evaluator.py):

```python
import json
from evaluation.evaluator import HinglishEvaluator

with open("eval_generations_100.json", "r", encoding="utf-8") as f:
    test_samples = json.load(f)

evaluator = HinglishEvaluator(
    judge_model_name="gemini-1.5-flash",
    judge_api_key="YOUR_GEMINI_API_KEY"
)

report = evaluator.evaluate_generation_quality(test_samples, run_llm_judge=True)
summary_md = evaluator.generate_markdown_summary(report)
print(summary_md)
```

---

### 4. Multi-GPU Distributed Training via CLI

Launch distributed training across multiple GPUs using Hugging Face `accelerate`:

```bash
# Set your Hugging Face Access Token
export HF_TOKEN="hf_your_access_token"

# Launch multi-GPU distributed SFT
accelerate launch lora/train.py --config lora/config.yaml
```

---

## 📊 Comprehensive Evaluation Suite

The `evaluation/` module provides a comprehensive 3-pillar benchmarking framework shared across all PEFT recipes:

```
                          EVALUATION SUITE
                         /        |       \
       1. QUANTITATIVE        2. SCRIPT           3. QUALITATIVE
         PERPLEXITY            PURITY             LLM-AS-A-JUDGE
       --------------        ----------           --------------
       • Teacher-forcing     • Latin char %       • Coherence (1-5)
       • PPL = exp(Loss)     • Devanagari bleed   • Helpfulness (1-5)
       • Fast forward-pass   • 100% Romanized %   • Naturalness (1-5)
       • Zero API costs      • Pure CPU regex     • Blind Win Rate (%)
```

### 🏆 Benchmark Metrics:
* **Perplexity Reduction:** Quantifies confidence gain on domain language ($PPL_{\text{reduction}} = \frac{PPL_{\text{base}} - PPL_{\text{lora}}}{PPL_{\text{base}}} \times 100$).
* **Script Purity Score:** Measures percentage of Latin characters vs accidental Devanagari character bleed.
* **LLM-as-a-Judge (Gemini / OpenAI):** 
  * *Coherence & Sensicality (1–5)*
  * *Helpfulness & Factuality (1–5)*
  * *Hinglish Naturalness & Code-Mixing (1–5)*
  * *Blind Head-to-Head Pairwise Win Rate (%)*

---

## ⚡ Speed Optimization & Hardware Tuning

| Parameter / Lever | 🐢 Default Setup | ⚡ Optimized Setup | Technical Reason |
|---|---|---|---|
| **Precision on P100 / T4** | `use_bf16: true` | `use_fp16: true, use_bf16: false` | Turing/Pascal lacks native BF16; emulated BF16 is 10x slower than FP16 Tensor Cores. |
| **Attention Implementation** | Eager Attention | `attn_implementation="sdpa"` | Scaled Dot-Product Attention fuses CUDA kernels, avoiding intermediate matrix allocation. |
| **Sequence Length ($L$)** | `max_seq_length: 2048` | `max_seq_length: 1024` | Attention FLOPs scale as $O(L^2)$. Conversational turns average ~250–350 tokens. |
| **Validation Strategy** | Full test set (4,500 samples) | `max_eval_samples: 200` | Avoids 35-minute evaluation stalls every 500 steps, cutting eval time to 15 seconds. |
| **Inference KV-Caching** | `use_cache: false` | `use_cache: true` | Prevents re-computing $O(N^2)$ past attention states during autoregressive token generation. |
| **Inference Batching** | `batch_size: 1` (~50s / pair) | `batch_size: 8` with left-padding (~3.5s / pair) | Slices generation time by ~15x by saturating GPU compute cores concurrently. |

---

## 🗺️ PEFT Recipes Roadmap

This repository is actively developing dedicated recipe implementations for each major PEFT paradigm:

- [x] **Recipe 1: LoRA (Low-Rank Adaptation):** Standard low-rank weight decomposition on LLaMA-3.2-1B.
- [x] **Recipe 2: QLoRA (Quantized LoRA):** 4-bit NormalFloat (NF4) base model quantization on Mistral-7B-Instruct-v0.3.
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
