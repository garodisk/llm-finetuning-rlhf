# LLM Fine-Tuning & Alignment: Technical Guide

This guide provides an in-depth walkthrough of two essential LLM post-training techniques: **Supervised Fine-Tuning (SFT)** and **RLHF via Direct Preference Optimization (DPO)**.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Part 1: Supervised Fine-Tuning (SFT)](#part-1-supervised-fine-tuning-sft)
4. [Part 2: RLHF with Direct Preference Optimization (DPO)](#part-2-rlhf-with-direct-preference-optimization-dpo)
5. [How SFT and DPO Work Together](#how-sft-and-dpo-work-together)
6. [Published Models](#published-models)
7. [Key Libraries & Concepts](#key-libraries--concepts)

---

## Introduction

### The Post-Training Pipeline

Modern LLM assistants like ChatGPT aren't just pre-trained on text—they go through additional training stages:

```
Pre-training (next token prediction on web text)
        │
        ▼
    [ SFT ]  →  Model learns to follow instructions
        │
        ▼
    [ RLHF/DPO ]  →  Model outputs align with human preferences
        │
        ▼
  Helpful, harmless assistant
```

This repository demonstrates both stages with working code.

### What Each Technique Does

| Technique | Input Data | What the Model Learns |
|-----------|------------|----------------------|
| **SFT** | (instruction, response) pairs | How to format helpful answers |
| **DPO** | (prompt, good_response, bad_response) triples | Which responses humans prefer |

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (T4, V100, A100, or RTX 3090/4090)
- **RAM**: 16GB+ system memory

### Software Requirements
```bash
Python >= 3.10
PyTorch >= 2.0
CUDA >= 11.8
```

### Installation
```bash
pip install transformers>=4.57 datasets>=4.0 peft>=0.18 trl>=0.26
pip install accelerate bitsandbytes evaluate
pip install tensorboard wandb
```

---

## Part 1: Supervised Fine-Tuning (SFT)

**Notebook**: `Supervised_Fine_Tuning_(SFT)_Finetuning.ipynb`

### What is SFT?

Supervised Fine-Tuning teaches a pre-trained LLM to follow instructions by training on (instruction, response) pairs. Before SFT, the model just predicts likely next tokens. After SFT, it generates helpful, structured responses.

**Example training pair:**
```
Instruction: "Explain quantum computing in simple terms"
Response: "Quantum computing uses quantum bits (qubits) that can be 0, 1,
          or both at once. This lets quantum computers solve certain
          problems much faster than regular computers..."
```

### The QLoRA Approach

Training a 7B model normally requires ~28GB VRAM. **QLoRA** makes it possible on a single 16GB GPU by combining:

1. **4-bit Quantization**: Compress base model weights from FP16 to NF4
2. **LoRA Adapters**: Train small matrices instead of all parameters

```
Standard fine-tuning:  ~28GB VRAM, 7B trainable params
QLoRA fine-tuning:     ~7GB VRAM,  40M trainable params (0.59%)
```

### Model & Dataset

| Component | Details |
|-----------|---------|
| Base Model | `meta-llama/Llama-2-7b-chat-hf` (7B parameters) |
| Dataset | `saketgarodia1/guanaco-llama2-chat-en` |
| Training Examples | 3,539 |
| Test Examples | 190 |
| Format | Chat-style instruction-response pairs |

### Key Configuration

#### 4-bit Quantization Config
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Use 4-bit weights
    bnb_4bit_quant_type="nf4",            # NormalFloat4 - best for transformers
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bf16
    bnb_4bit_use_double_quant=True,       # Quantize the quantization scales too
)
```

**Why NF4?** NormalFloat4 places quantization buckets according to the normal distribution of transformer weights, giving lower error than uniform 4-bit quantization.

#### LoRA Configuration
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                    # Rank of adapter matrices (higher = more capacity)
    lora_alpha=32,           # Scaling factor (alpha/r determines learning rate scaling)
    lora_dropout=0.05,       # Dropout for regularization
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # FFN
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Why these target modules?** Attention projections (q, k, v, o) control how the model attends to context. FFN projections (gate, up, down) control knowledge retrieval. Adapting both gives the best results.

#### Training Configuration
```python
from trl import SFTConfig

training_args = SFTConfig(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,      # Effective batch size = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",         # Smooth decay
    warmup_ratio=0.03,                  # 3% warmup steps
    max_seq_length=1024,
    optim="paged_adamw_32bit",          # Memory-efficient optimizer
    bf16=True,                          # Mixed precision
    gradient_checkpointing=True,        # Trade compute for memory
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
)
```

**Key hyperparameters explained:**
- `gradient_accumulation_steps=8`: With batch_size=1, this gives effective batch of 8
- `gradient_checkpointing=True`: Recomputes activations during backward pass to save memory
- `paged_adamw_32bit`: Offloads optimizer states to CPU when GPU memory is tight

### Training Results

| Metric | Before Training | After Training | Improvement |
|--------|-----------------|----------------|-------------|
| Eval Loss | 2.42 | 1.19 | -51% |
| Perplexity | 11.24 | 3.28 | -71% |

### Evaluation with GPT-4 as Judge

The notebook includes LLM-as-judge evaluation—using GPT-4 to score model outputs:

```python
# Evaluation criteria:
rubric = """
Rate the response on:
1. Correctness (0-10): Is it factually accurate?
2. Helpfulness (0-10): Does it address the user's needs?
3. Clarity (0-10): Is it well-structured and easy to understand?
4. Instruction Adherence (0-10): Does it follow the prompt?
5. Politeness (0-10): Is the tone appropriate?
"""
```

The evaluation tracks:
- Average scores across 50 test examples
- Hallucination rate
- Format violations
- Safety concerns

### Output Artifacts

- **LoRA Adapter saved to**: Google Drive + HuggingFace Hub
- **Model ID**: `saketgarodia1/llama2-7b-guanaco-qlora-adapter`
- **Checkpoint**: checkpoint-400

---

## Part 2: RLHF with Direct Preference Optimization (DPO)

**Notebook**: `RLHF_with_DPO.ipynb`

### What is DPO?

Direct Preference Optimization aligns model outputs with human preferences **without training a separate reward model**. Traditional RLHF requires:

1. Train a reward model on preference data
2. Use RL (PPO) to optimize the LLM against the reward model

DPO simplifies this to a single supervised training step.

### DPO vs Traditional RLHF

| Aspect | Traditional RLHF (PPO) | DPO |
|--------|------------------------|-----|
| Reward Model | Required (separate training) | Not needed |
| Training Stability | Can be unstable | Stable |
| Hyperparameter Sensitivity | High | Low |
| Implementation Complexity | Complex | Simple |
| Compute Requirements | High | Lower |

### How DPO Works

Given a prompt and two responses (chosen and rejected), DPO directly optimizes:

```
Loss = -log(σ(β * (log π(chosen) - log π(rejected) - log π_ref(chosen) + log π_ref(rejected))))
```

In plain English: **Make the model more likely to generate "chosen" responses and less likely to generate "rejected" responses**, while staying close to the original model.

### Model & Dataset

| Component | Details |
|-----------|---------|
| Base Model | `teknium/OpenHermes-2.5-Mistral-7B` |
| Dataset | `Intel/orca_dpo_pairs` |
| Training Examples | 12,859 preference pairs |
| Format | {system, question, chosen, rejected} |

### Preference Pair Example

```json
{
  "system": "You are a helpful AI assistant.",
  "question": "What causes rain?",
  "chosen": "Rain forms when water vapor in the atmosphere condenses into
             droplets. This happens when moist air rises, cools, and can
             no longer hold all its moisture. The droplets combine and
             fall as precipitation when they become heavy enough.",
  "rejected": "Rain is water falling from clouds. It happens because of
              weather and stuff. Clouds get heavy and rain falls down."
}
```

The model learns that detailed, accurate responses are preferred over vague ones.

### ChatML Format

The notebook converts data to ChatML format for Mistral:

```
<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
What causes rain?<|im_end|>
<|im_start|>assistant
Rain forms when water vapor...<|im_end|>
```

```python
def format_chat(example):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["chosen"]}  # or rejected
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)
```

### DPO Configuration

```python
from trl import DPOConfig

dpo_config = DPOConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # Effective batch = 16
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    warmup_steps=50,
    beta=0.1,                           # DPO temperature
    max_prompt_length=1024,
    max_length=1536,
    bf16=True,
    gradient_checkpointing=True,
)
```

### The Beta Parameter

`beta` controls how much the model can deviate from the reference (base) model:

| Beta Value | Effect |
|------------|--------|
| Low (0.1) | More deviation allowed, stronger preference learning |
| Medium (0.2-0.3) | Balanced |
| High (0.5+) | Stays closer to base model, conservative updates |

Lower beta = more aggressive preference optimization, but risk of "reward hacking"
Higher beta = safer but slower learning

### LoRA for DPO

Same LoRA approach as SFT:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
```

### Training

```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=None,           # DPOTrainer handles reference model internally
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

trainer.train()
```

**Final training loss**: 0.040

### Inference Example

```python
from transformers import pipeline

pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain black holes simply."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
output = pipe(prompt, max_new_tokens=200, temperature=0.7, top_p=0.9)
```

### Output Artifacts

- **LoRA Adapter**: `final_dpo_adapter/`
- **Merged Model**: `saketgarodia1/OpenHermes-2.5-Mistral-7B-DPO`

---

## How SFT and DPO Work Together

In practice, these techniques are used in sequence:

### Stage 1: SFT
- **Input**: Pre-trained base model + instruction dataset
- **Output**: Model that can follow instructions
- **What it learns**: Format, structure, helpfulness

### Stage 2: DPO
- **Input**: SFT model + preference dataset
- **Output**: Model aligned with human preferences
- **What it learns**: Quality, safety, style preferences

### Why Both?

| If you only do... | Problem |
|-------------------|---------|
| SFT only | Model follows instructions but may give verbose, unhelpful, or unsafe responses |
| DPO only | Model has preferences but doesn't know how to structure responses |
| SFT → DPO | Model follows instructions AND outputs align with preferences |

### Real-World Pipeline (e.g., ChatGPT)

```
GPT-4 Base (pre-trained)
    │
    ▼
SFT on instruction data (100k+ examples)
    │
    ▼
RLHF/DPO on preference data (human rankings)
    │
    ▼
Safety fine-tuning (red-teaming data)
    │
    ▼
ChatGPT
```

---

## Published Models

| Model | Technique | Base | Link |
|-------|-----------|------|------|
| Llama-2 QLoRA Adapter | SFT | Llama-2-7B-Chat | [HuggingFace](https://huggingface.co/saketgarodia1/llama2-7b-guanaco-qlora-adapter) |
| Mistral DPO | DPO | OpenHermes-2.5-Mistral-7B | [HuggingFace](https://huggingface.co/saketgarodia1/OpenHermes-2.5-Mistral-7B-DPO) |

---

## Key Libraries & Concepts

### Libraries

| Library | Purpose |
|---------|---------|
| `transformers` | Model loading, tokenization, training |
| `peft` | LoRA, QLoRA adapters |
| `trl` | SFTTrainer, DPOTrainer |
| `bitsandbytes` | 4-bit/8-bit quantization |
| `accelerate` | Multi-GPU, mixed precision |

### Key Concepts

#### QLoRA
Quantized Low-Rank Adaptation. Keeps base model in 4-bit, trains small FP16 adapter matrices.

#### Gradient Checkpointing
Recomputes activations during backward pass instead of storing them. Saves ~50% memory at ~20% speed cost.

#### Gradient Accumulation
Simulates larger batch sizes: `effective_batch = per_device_batch × accumulation_steps`

#### ChatML
Chat Markup Language. Standard format for instruction models using special tokens like `<|im_start|>` and `<|im_end|>`.

#### Reference Model (in DPO)
The original model before DPO training. DPO loss includes a KL penalty to prevent the model from deviating too far from this reference.

---

## Conclusion

This repository demonstrates the two key post-training stages for building LLM assistants:

1. **SFT** teaches the model to follow instructions using QLoRA for efficiency
2. **DPO** aligns outputs with human preferences without complex RL

Together, they transform a raw pre-trained model into a helpful, aligned assistant.

Happy training!
