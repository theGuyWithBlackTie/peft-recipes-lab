# 🧪 PEFT Recipes Lab: Medical Chain-of-Thought (CoT)

<div align="center">

[![Branch: cot](https://img.shields.io/badge/Branch-cot-blue.svg?logo=git&logoColor=white)](https://github.com/theGuyWithBlackTie/peft-recipes-lab/tree/cot)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face PEFT](https://img.shields.io/badge/%F0%9F%A4%97%20PEFT-0.13%2B-yellow)](https://github.com/huggingface/peft)
[![TRL SFTTrainer](https://img.shields.io/badge/TRL-SFTTrainer-orange)](https://github.com/huggingface/trl)
[![Dataset: medical-cot](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-medical--cot-green)](https://huggingface.co/datasets/theguywithblacktie/medical-cot)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hardware: Kaggle P100 / T4 x 2](https://img.shields.io/badge/Hardware-Kaggle%20P100%20%7C%20T4%20x%202-76B900.svg?logo=nvidia&logoColor=white)](https://www.nvidia.com/)

**Production-grade, modular, and reproducible PEFT fine-tuning recipes for DeepSeek-R1-style Long-Form Chain-of-Thought (CoT) reasoning and clinical diagnosis.**

[Overview](#-overview--mission) • [Branch Evolution](#-differences-between-main-and-cot-branches) • [Experimental Ladder](#-the-4-tier-experimental-ladder) • [Evaluation Suite](#-dual-pillar-clinical-evaluation-suite) • [Quickstart](#-quickstart) • [Kaggle Execution](#-interactive-kaggle-execution)

</div>

---

## 🎯 Overview & Mission

The **`cot`** branch transforms the repository's PEFT pipeline from conversational instruction-following into a state-of-the-art **Chain-of-Thought (CoT) reasoning framework**, tailored for complex clinical diagnosis and biomedical deduction.

Standard fine-tuning often trains models to leap directly to conclusions, leading to diagnostic hallucinations and missed contraindications. Here, we fine-tune **Mistral-7B-Instruct-v0.3** using **4-bit QLoRA** with an explicit **DeepSeek-R1-style internal monologue** (`<think> ... </think>`) to formulate hypotheses, analyze evidence, resolve contradictions, and self-correct before giving a final diagnosis.

---

## 🔄 Differences Between `main` and `cot` Branches

| Dimension | `main` Branch (Hinglish Chat) | `cot` Branch (Medical CoT Reasoning) |
| :--- | :--- | :--- |
| **Domain & Task** | Romanized Hindi-English conversational assistant | Complex clinical diagnostics & pathophysiology deduction |
| **Reasoning Paradigm** | Direct response generation | **DeepSeek-R1-style** `<think> ... </think>` scratchpad reflection |
| **Dataset** | `theguywithblacktie/hinglish-conversations` | [`theguywithblacktie/medical-cot`](https://huggingface.co/datasets/theguywithblacktie/medical-cot) (1,800 train / 200 test) |
| **Loss Masking** | Uniform SFT on assistant turns | Completion masking on `[/INST]` + **Progressive Per-Token Loss Weighting** |
| **Evaluation Pillar A** | Latin vs. Devanagari Script Purity | **Structural & Cognitive Metrics**: format compliance, thought length, reflection marker density |
| **Evaluation Pillar B** | Hinglish naturalness & tone judge | **4-Axis Clinical Judge**: intermediate hypotheses, evidence utilization, contradiction handling, self-reflection |
| **Kaggle Notebook** | `lora/finetune-mistral-7b.ipynb` | [`lora/finetune-mistral-7b-cot.ipynb`](lora/finetune-mistral-7b-cot.ipynb) with git-pull sync & multi-GPU support |

---

## 🪜 The 4-Tier Experimental Ladder

To rigorously measure what reasoning tokens and differential supervision contribute to clinical performance, the pipeline implements a 4-tier experimental ladder:

```
Tier 0: Zero-Shot Base       ──> Evaluates un-finetuned Mistral-7B out-of-the-box
        │
Tier 1: Direct SFT Baseline  ──> Direct diagnosis (no <think> scratchpad)
        │
Tier 2: Standard CoT SFT     ──> Full <think>...</think> reflection with uniform loss
        │
Tier 3: Progressive Weighted ──> Differential token supervision:
        Loss SFT                 <think>: 0.25 | reasoning: 0.50 | </think>: 0.75 | answer: 1.00
```

### ⚖️ Tier 3 Progressive Token-Weighting Schedule

During Tier 3, `ProgressiveWeightedLossTrainer` and `ProgressiveWeightedLossCollator` apply differential loss pressure:

$$\mathcal{L}_{\text{weighted}} = \frac{\sum_{t=1}^{T} w_t \cdot \ell_{\text{CE}}(y_t, \hat{y}_t)}{\sum_{t=1}^{T} w_t}$$

| Token Category | Weight ($w_t$) | Rationale & Mechanism |
| :--- | :---: | :--- |
| **Prompt / Instruction** | `0.00` | Masked with label `-100`; no loss on user questions. |
| **`<think>` trigger tag** | `0.25` | Low pressure to enter the scratchpad naturally. |
| **`reasoning_tokens` (internal thought)** | `0.50` | Moderate supervision; encourages differential exploration and backtracking without penalizing intermediate dead ends. |
| **`</think>` closing tag** | `0.75` | Higher pressure to cleanly conclude thinking and transition to user response. |
| **`answer_tokens` (clinical diagnosis)** | `1.00` | Full supervision pressure on diagnostic accuracy, safety, and intervention planning. |

---

## 📊 Dual-Pillar Clinical Evaluation Suite

The evaluation suite ([`evaluation/`](evaluation/)) provides a unified benchmark across all 4 experimental tiers on the fixed 200 held-out test cases:

```
                               EVALUATION SUITE
                              /                \
          PILLAR A: STRUCTURAL METRICS      PILLAR B: CLINICAL LLM-AS-A-JUDGE
          ----------------------------      ---------------------------------
          • <think> format compliance (%)   • 1. Intermediate Diagnostic Hypotheses (1-5)
          • Tag integrity (unclosed tags)   • 2. Evidence Utilization (1-5)
          • Average thought token length    • 3. Contradiction Handling (1-5)
          • Reasoning-to-Answer ratio       • 4. Reflection & Self-Correction (1-5)
          • Reflection marker density       • Diagnostic accuracy & clinical safety
```

### 🔬 Pillar B Clinical Reasoning Axes:
1. **Intermediate Diagnostic Hypotheses (1–5):** Does the model explore differential diagnostic possibilities rather than jumping blindly to conclusions?
2. **Evidence Utilization (1–5):** How thoroughly does the model connect the patient's symptoms, vitals, history, and examination findings?
3. **Contradiction Handling (1–5):** How effectively does the model identify conflicting signs, resolve ambiguity, and rule out competing conditions?
4. **Reflection & Self-Correction (1–5):** Does the model demonstrate internal cognitive auditing (*"Wait, let me double-check this mechanism..."*)?

---

## 📂 Repository Structure

```tree
peft-recipes-lab/
├── dataset/
│   ├── prepare_cot_dataset.py       # Partitions ty-kim/medical_cot into 1,800 train / 200 test & pushes to Hub
│   └── README.md                    # Detailed dataset card published to the Hugging Face Hub
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
│   ├── config.yaml                  # Mistral-7B QLoRA configuration (P100 FP16, max_seq 2048, loss weights)
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
└── README.md                        # Documentation for cot branch
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone -b cot https://github.com/theGuyWithBlackTie/peft-recipes-lab.git
cd peft-recipes-lab

# Install dependencies (trl<0.15.0 ensures multi-GPU stability)
pip install -q "transformers>=4.46.0" "peft>=0.13.0" "trl<0.15.0" "accelerate>=0.34.0" "bitsandbytes>=0.43.0" datasets pyyaml huggingface_hub matplotlib
```

### 2. Dataset Preparation & Hub Sync

The dataset is partitioned into 1,800 training and 200 test samples:

```bash
# Partitions ty-kim/medical_cot and exports to dataset/hf_export/medical_cot
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

Open [`lora/finetune-mistral-7b-cot.ipynb`](lora/finetune-mistral-7b-cot.ipynb) on Kaggle with GPU accelerator (**P100** or **T4 x 2**):

1. **Cell 1** automatically pulls the latest code from GitHub branch `cot`.
2. **Cell 4** lets you choose your experiment mode:
   ```python
   EXPERIMENT_MODE = "cot"  # Options: 'cot', 'direct', or 'weighted_cot'
   ```
3. **Cell 7** executes training, saves adapters, and uploads to Hugging Face Hub.
4. **Cell 8** runs test clinical inference displaying the model's `<think>...</think>` reflection and diagnosis!
