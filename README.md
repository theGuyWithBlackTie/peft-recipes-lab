# 🧪 PEFT Recipes Lab: Medical Chain-of-Thought (CoT)

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face PEFT](https://img.shields.io/badge/%F0%9F%A4%97%20PEFT-0.13%2B-yellow)](https://github.com/huggingface/peft)
[![TRL SFTTrainer](https://img.shields.io/badge/TRL-SFTTrainer-orange)](https://github.com/huggingface/trl)
[![Dataset: medical-cot](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-medical--cot-green)](https://huggingface.co/datasets/theguywithblacktie/medical-cot)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hardware: Kaggle P100 / T4 x 2 / A100](https://img.shields.io/badge/Hardware-Kaggle%20P100%20%7C%20T4%20x%202%20%7C%20A100-76B900.svg?logo=nvidia&logoColor=white)](https://www.nvidia.com/)

**Production-grade, modular, and reproducible Parameter-Efficient Fine-Tuning (PEFT) recipes for DeepSeek-R1-style Long-Form Chain-of-Thought (CoT) reasoning and clinical diagnosis.**

[Overview](#-overview--mission) • [Architecture & Hardware](#-architecture--hardware-optimization) • [Dataset](#-dataset--curation) • [The 4-Tier Experimental Ladder](#-the-4-tier-experimental-ladder) • [Evaluation Suite](#-dual-pillar-clinical-evaluation-suite) • [Repository Structure](#-repository-structure) • [Quickstart](#-quickstart) • [Kaggle Execution](#-interactive-kaggle-execution)

</div>

---

## 🎯 Overview & Mission

**`PEFT Recipes Lab`** is an open-source engineering lab dedicated to practical, high-throughput Parameter-Efficient Fine-Tuning (PEFT) recipes for modern Large Language Models.

This recipe focuses on teaching open-weights language models (**Mistral-7B-Instruct-v0.3**) to perform **DeepSeek-R1-style Long-Form Chain-of-Thought (CoT)** reasoning for complex clinical diagnosis and pathophysiology deduction.

### 💡 Why Chain-of-Thought in Clinical Reasoning?
Standard instruction fine-tuning teaches models to output immediate answers. In medicine, jumping straight to a diagnosis often leads to hallucinations, missed contraindications, and confirmation bias. By training models with an explicit internal scratchpad (`<think> ... </think>`), the model learns to:
1. **Formulate Differential Hypotheses:** Brainstorm plausible diagnostic possibilities before committing.
2. **Utilize Subtle Evidence:** Connect patient vitals, timeline, physical findings, and travel/medication history.
3. **Handle Contradictions:** Identify conflicting clinical signs and rule out competing conditions with medical justification.
4. **Self-Correct & Backtrack:** Re-evaluate premature deductions (*"Wait, let me double-check if that explains the edema..."*) prior to delivering the final diagnosis and management plan.

---

## ⚡ Architecture & Hardware Optimization

This recipe is engineered to run on commodity cloud GPUs (such as **Kaggle NVIDIA P100 16GB** and **Kaggle GPU T4 x 2 15GB**) with zero out-of-memory (OOM) errors:

* **4-bit NormalFloat (NF4) QLoRA:** Quantizes Mistral-7B's base weights into 4-bit with double quantization, reducing base model memory footprint to **~5.5 GB VRAM**.
* **Target LoRA Modules:** Adapts all linear attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with rank $r=16, \alpha=32$.
* **Hardware Precision Routing:** Automatically checks for native hardware `bfloat16` support (`torch.cuda.is_bf16_supported()`). Routes to native FP16 Tensor Cores on Pascal (P100) and Turing (T4) GPUs to avoid slow software emulation, while enabling BF16 on Ampere/Hopper (A100/H100).
* **Multi-GPU / Single-GPU Stability:** Pinned dependency matrix (`trl<0.15.0`) prevents cross-device Triton pointer mismatch on sharded multi-GPU configurations (`GPU T4 x 2`).
* **Long Context Window:** Configured with `max_seq_length=2048` to comfortably accommodate rich internal reasoning monologues alongside the final diagnosis.
* **Completion-Only Loss Masking:** Masks user prompts (`[INST] ... [/INST]`) with label `-100`, ensuring cross-entropy loss is strictly calculated over assistant completions.

---

## 📦 Dataset & Curation

The recipe utilizes the curated [`theguywithblacktie/medical-cot`](https://huggingface.co/datasets/theguywithblacktie/medical-cot) dataset on Hugging Face (derived and partitioned from `ty-kim/medical_cot`):

* **Total Samples:** 2,000 clinically rich medical problem scenarios.
* **Train Split:** 1,800 samples (90%) for supervised fine-tuning.
* **Test Split:** 200 samples (10%) strictly held-out for unbiased benchmarking across all experimental tiers.
* **Seed:** `42` (deterministic partition).

### Schema Features:
* `Question`: Clinical vignette, patient presentation, vitals, and diagnostic inquiry.
* `Complex_CoT`: Detailed internal reasoning steps, differential exploration, and self-reflection.
* `Response`: Authoritative final clinical diagnosis, rationale, and recommended intervention.
* `text_cot`: Pre-formatted for R1-style fine-tuning (`[INST] ... [/INST] <think>\n...\n</think>\n...`).
* `text_direct`: Pre-formatted for direct baseline fine-tuning (`[INST] ... [/INST] ...`).

---

## 🪜 The 4-Tier Experimental Ladder

To scientifically measure the exact capability gains attributable to reasoning tokens and differential supervision, this repository implements a 4-tier experimental ladder:

```
Tier 0: Zero-Shot Base       ──> Evaluates un-finetuned Mistral-7B out-of-the-box
        │
Tier 1: Direct SFT Baseline  ──> Direct diagnosis without reasoning scratchpad (text_direct)
        │
Tier 2: Standard CoT SFT     ──> Full <think>...</think> reflection with uniform loss (text_cot)
        │
Tier 3: Progressive Weighted ──> Differential per-token supervision:
        Loss SFT                 <think>: 0.25 | reasoning: 0.50 | </think>: 0.75 | answer: 1.00
```

### ⚖️ Tier 3 Progressive Loss-Weighting Schedule

During Tier 3, `ProgressiveWeightedLossTrainer` and `ProgressiveWeightedLossCollator` dynamically scan token sequences following `[/INST]` and assign per-token loss weights:

$$\mathcal{L}_{\text{weighted}} = \frac{\sum_{t=1}^{T} w_t \cdot \ell_{\text{CE}}(y_t, \hat{y}_t)}{\sum_{t=1}^{T} w_t}$$

| Token Category | Weight ($w_t$) | Rationale & Mechanism |
| :--- | :---: | :--- |
| **Prompt / Instruction** | `0.00` | Masked with label `-100`; model is not penalized for user prompts. |
| **`<think>` trigger tag** | `0.25` | Low supervision pressure to enter the scratchpad naturally. |
| **`reasoning_tokens` (internal thought)** | `0.50` | Moderate supervision; encourages differential exploration and backtracking without penalizing exploratory wording or intermediate dead ends. |
| **`</think>` closing tag** | `0.75` | Higher pressure to ensure the model cleanly concludes internal thinking and transitions to user-facing output. |
| **`answer_tokens` (clinical response)** | `1.00` | Full supervision pressure on diagnostic accuracy, safety, and intervention planning. |

---

## 📊 Dual-Pillar Clinical Evaluation Suite

The evaluation suite ([`evaluation/`](evaluation/)) provides a unified benchmarking engine across all 4 experimental tiers on the fixed 200 held-out test cases:

```
                               EVALUATION SUITE
                              /                \
          PILLAR A: STRUCTURAL METRICS      PILLAR B: CLINICAL LLM-AS-A-JUDGE
          ----------------------------      ---------------------------------
          • <think> format compliance (%)   • 1. Intermediate Diagnostic Hypotheses (1-5)
          • Tag integrity (unclosed tags)   • 2. Evidence Utilization (1-5)
          • Average thought token length    • 3. Contradiction Handling (1-5)
          • Reasoning-to-Answer ratio       • 4. Reflection & Self-Correction (1-5)
          • Reflection marker density       • Diagnostic accuracy & safety verdict
```

### 🔍 Pillar A: Structural & Cognitive Metrics ([`evaluation/cot_evaluator.py`](evaluation/cot_evaluator.py))
* **Format Compliance Rate (%):** Percentage of generations with properly opened `<think>` and closed `</think>` tags.
* **Reasoning vs. Answer Length:** Measures token count inside `<think>...</think>` vs. final answer.
* **Reasoning Ratio:** Ratio of internal thinking tokens to total generated tokens.
* **Reflection Marker Density:** Regex-based frequency per 1,000 tokens of cognitive words (*"wait"*, *"let me recheck"*, *"however"*, *"alternatively"*, *"rule out"*, *"unlikely"*).

### 🩺 Pillar B: Clinical LLM-as-a-Judge ([`evaluation/clinical_judge.py`](evaluation/clinical_judge.py))
Evaluates clinical rigor against reference diagnoses along 4 dedicated axes (scored 1–5):
1. **Intermediate Diagnostic Hypotheses (1–5):** Does the model explore differential diagnostic possibilities rather than jumping blindly to conclusions?
2. **Evidence Utilization (1–5):** How thoroughly does the model connect the patient's symptoms, vitals, history, and examination findings?
3. **Contradiction Handling (1–5):** How effectively does the model identify conflicting signs, resolve ambiguity, and rule out competing conditions?
4. **Reflection & Self-Correction (1–5):** Does the model demonstrate internal cognitive auditing (*"Wait, let me double-check this mechanism..."*)?

### 📈 Comparative Ablation Reports ([`evaluation/run_benchmark.py`](evaluation/run_benchmark.py))
Running the benchmark automatically updates a side-by-side comparative Markdown table in `lora/eval_results/cot_ablation_report.md` comparing Tiers 0, 1, 2, and 3.

---

## 📂 Repository Structure

```tree
peft-recipes-lab/
├── dataset/
│   ├── prepare_cot_dataset.py       # Partitions ty-kim/medical_cot into 1,800 train / 200 test & pushes to Hub
│   └── README.md                    # Detailed dataset card documentation
│
├── docs/
│   ├── reasoning_styles_guide.md    # Guide to the 4 CoT reasoning paradigms (R1, Math, Logic, Code)
│   └── superpowers/
│       ├── specs/                   # System architecture and 4-tier experimental design spec
│       └── plans/                   # Test-driven implementation plan
│
├── evaluation/
│   ├── cot_evaluator.py             # Pillar A: Rule-based <think> metrics, token stats, reflection density
│   ├── clinical_judge.py            # Pillar B: 4-axis LLM judge (Gemini / OpenAI) for clinical deduction
│   ├── run_benchmark.py             # Unified CLI runner comparing Tiers 0, 1, 2, and 3
│   └── perplexity.py                # Cross-entropy loss & perplexity evaluator
│
├── lora/
│   ├── config.yaml                  # Mistral-7B QLoRA configuration (P100/T4 FP16, max_seq 2048, loss weights)
│   ├── finetune-mistral-7b-cot.ipynb# Kaggle notebook with git-pull sync & multi-GPU support
│   ├── train.py                     # CLI training runner supporting --mode {direct,cot,weighted_cot}
│   └── src/
│       ├── data_loader.py           # Mode-aware data loader (text_cot vs. text_direct) with Hub/disk fallback
│       ├── model_builder.py         # 4-bit QLoRA builder with BitsAndBytes NF4 & FP16 compute
│       ├── trainer.py               # SFTTrainer wrapper with Kaggle P100 / T4 memory optimizations
│       ├── weighted_loss_trainer.py # Progressive per-token loss collator & weighted trainer (Tier 3)
│       └── inference.py             # Long-form generation with clean stopping criteria
│
├── utilities/
│   └── plotting.py                  # Training loss and perplexity dashboard plotter
│
├── Pipfile                          # Pinned dependencies (trl<0.15.0 for stable multi-GPU training)
└── README.md                        # Master repository documentation
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/theGuyWithBlackTie/peft-recipes-lab.git
cd peft-recipes-lab

# Install dependencies (trl<0.15.0 ensures multi-GPU stability)
pip install -q "transformers>=4.46.0" "peft>=0.13.0" "trl<0.15.0" "accelerate>=0.34.0" "bitsandbytes>=0.43.0" datasets pyyaml huggingface_hub matplotlib
```

### 2. Dataset Preparation & Hub Sync

The dataset is partitioned into 1,800 training and 200 held-out test samples:

```bash
# Partitions and exports locally to dataset/hf_export/medical_cot
python dataset/prepare_cot_dataset.py

# Push directly to your Hugging Face Hub account:
python dataset/prepare_cot_dataset.py --push_to_hub --repo_id theguywithblacktie/medical-cot
```

### 3. Training Across Experimental Tiers

You can train any tier using the CLI runner by passing the `--mode` flag:

```bash
# Tier 1: Baseline Direct SFT (Question -> direct Response)
python lora/train.py --mode direct

# Tier 2: Standard CoT SFT (Question -> <think>...</think> -> Response)
python lora/train.py --mode cot

# Tier 3: Progressive Weighted-Loss CoT SFT
python lora/train.py --mode weighted_cot
```

### 4. Running Benchmarks & Comparative Ablations

Evaluate any model tier against the 200 held-out test cases:

```bash
# Evaluate Tier 0: Zero-Shot Base Mistral-7B
python evaluation/run_benchmark.py --tier 0 --model_id mistralai/Mistral-7B-Instruct-v0.3

# Evaluate Tier 1: Direct SFT Baseline
python evaluation/run_benchmark.py --tier 1 --adapter_path ./mistral-direct-lora-outputs

# Evaluate Tier 2: Standard CoT SFT
python evaluation/run_benchmark.py --tier 2 --adapter_path ./mistral-cot-lora-outputs

# Evaluate Tier 3: Progressive Weighted Loss SFT
python evaluation/run_benchmark.py --tier 3 --adapter_path ./mistral-weighted_cot-lora-outputs
```

*All metrics and side-by-side comparative Markdown tables are automatically aggregated in `lora/eval_results/cot_ablation_report.md`.*

---

## 📓 Interactive Kaggle Execution

Open [`lora/finetune-mistral-7b-cot.ipynb`](lora/finetune-mistral-7b-cot.ipynb) on Kaggle with a GPU accelerator (**P100** or **T4 x 2**):

1. **Cell 1** automatically pulls or clones the latest code into `/kaggle/working/peft-recipes-lab`.
2. **Cell 4** lets you choose your experiment mode:
   ```python
   EXPERIMENT_MODE = "cot"  # Options: 'cot', 'direct', or 'weighted_cot'
   ```
3. **Cell 7** executes training, saves adapters, and uploads to Hugging Face Hub via `trainer.train_and_save()`.
4. **Cell 8** runs test clinical inference displaying the model's `<think>...</think>` reflection and final diagnosis!
