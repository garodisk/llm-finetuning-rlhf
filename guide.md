# LLM Training Techniques: Complete Technical Guide

This guide provides an in-depth walkthrough of three essential LLM training paradigms demonstrated in this repository: **Supervised Fine-Tuning (SFT)**, **Knowledge Distillation**, and **Reinforcement Learning from Human Feedback (RLHF) via Direct Preference Optimization (DPO)**.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Part 1: Supervised Fine-Tuning (SFT)](#part-1-supervised-fine-tuning-sft)
4. [Part 2: Knowledge Distillation](#part-2-knowledge-distillation)
5. [Part 3: RLHF with Direct Preference Optimization (DPO)](#part-3-rlhf-with-direct-preference-optimization-dpo)
6. [Published Models & Datasets](#published-models--datasets)
7. [Key Libraries & Concepts](#key-libraries--concepts)
8. [Recommended Learning Path](#recommended-learning-path)

---

## Introduction

### What This Repository Teaches

Modern LLM development involves more than just pre-training. To create useful, efficient, and aligned models, practitioners use three complementary techniques:

| Technique | Purpose | When to Use |
|-----------|---------|-------------|
| **Supervised Fine-Tuning (SFT)** | Teach models to follow instructions | When you have instruction-response pairs |
| **Knowledge Distillation** | Compress large models into smaller ones | When you need efficiency for deployment |
| **DPO (RLHF)** | Align outputs with human preferences | When quality ranking data is available |

This repository provides hands-on implementations of all three, complete with real datasets, training code, and evaluation.

### What You'll Learn

- How to fine-tune a 7B parameter model on a single GPU using QLoRA
- How to distill knowledge from BERT to DistilBERT for 4x memory reduction
- How to apply post-training quantization (FP16, INT4) with minimal accuracy loss
- How to train preference-aligned models using DPO instead of PPO
- Best practices for evaluation, including LLM-as-judge techniques

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (T4, V100, A100, or RTX 3090/4090)
- **RAM**: 16GB+ system memory
- **Storage**: 20GB+ for model checkpoints

### Software Requirements
```bash
Python >= 3.10
PyTorch >= 2.0
CUDA >= 11.8
```

### Core Libraries
```bash
pip install transformers>=4.57 datasets>=4.0 peft>=0.18 trl>=0.26
pip install accelerate bitsandbytes evaluate scikit-learn
pip install seaborn matplotlib tensorboard wandb
```

---

## Part 1: Supervised Fine-Tuning (SFT)

**Notebook**: `Supervised_Fine_Tuning_(SFT)_Finetuning.ipynb`

### What is SFT?

Supervised Fine-Tuning adapts a pre-trained language model to follow instructions by training it on (instruction, response) pairs. The model learns to generate helpful, task-specific outputs instead of just predicting the next token.

### The QLoRA Approach

This notebook uses **QLoRA** (Quantized Low-Rank Adaptation), which combines:

1. **4-bit Quantization**: Compress the base model from FP16 to NF4 format, reducing memory by 4x
2. **LoRA Adapters**: Train small adapter matrices instead of all 7B parameters

```
Memory without QLoRA: ~28GB (FP16)
Memory with QLoRA:    ~7GB  (4-bit + adapters)
```

### Model & Dataset

| Component | Details |
|-----------|---------|
| Base Model | `meta-llama/Llama-2-7b-chat-hf` (7B parameters) |
| Dataset | `saketgarodia1/guanaco-llama2-chat-en` (3,539 training examples) |
| Format | Chat-style instruction-response pairs |

### Key Configuration

#### BitsAndBytes Quantization Config
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 - optimal for transformers
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # Quantize the quantization constants
)
```

#### LoRA Configuration
```python
lora_config = LoraConfig(
    r=16,                    # Rank of adapter matrices
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.05,
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Trainable Parameters**: ~40M (0.59% of 6.8B total)

#### Training Arguments
```python
SFTConfig(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,      # Effective batch size = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_seq_length=1024,
    optim="paged_adamw_32bit",
    bf16=True,
    gradient_checkpointing=True,
)
```

### Results

| Metric | Before Training | After Training | Improvement |
|--------|-----------------|----------------|-------------|
| Eval Loss | 2.42 | 1.19 | -51% |
| Perplexity | 11.24 | 3.28 | -71% |

### Evaluation with GPT-4 as Judge

The notebook includes LLM-as-judge evaluation using GPT-4:

```python
# Evaluation criteria:
# - Correctness: Is the answer factually accurate?
# - Helpfulness: Does it address the user's needs?
# - Clarity: Is the response well-structured?
# - Instruction Adherence: Does it follow the prompt?
# - Politeness: Is the tone appropriate?
```

This approach evaluates 50 test examples and tracks:
- Hallucination rate
- Format violations
- Safety concerns

### Output Artifacts

- **LoRA Adapter**: Saved to Google Drive and pushed to HuggingFace
- **Model ID**: `saketgarodia1/llama2-7b-guanaco-qlora-adapter`

---

## Part 2: Knowledge Distillation

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model, achieving similar performance with significantly reduced compute requirements.

### Notebook Progression

| Notebook | Purpose | Output |
|----------|---------|--------|
| `Distillation_part_1.ipynb` | Dataset preparation | Stratified IT ticket dataset |
| `Distillation_part_2.ipynb` | Teacher model training | BERT classifier (88% acc) |
| `Distillation_part_2_train_pooler_only.ipynb` | Alternative: freeze encoder | Pooler-only training |
| `Distilliation_Part_IV.ipynb` | Quantization comparison | FP32 vs FP16 vs INT4 |

---

### Part 2A: Dataset Preparation

**Notebook**: `Distillation_part_1.ipynb`

#### Dataset Overview

| Property | Value |
|----------|-------|
| Source | IT Service Tickets |
| Total Samples | 47,837 |
| Classes | 8 categories |
| Task | Multi-class text classification |

#### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Hardware | 13,617 | 28.4% |
| HR Support | 10,915 | 22.8% |
| Access | 7,125 | 14.9% |
| Miscellaneous | 7,060 | 14.7% |
| Storage | 2,777 | 5.8% |
| Purchase | 2,464 | 5.1% |
| Internal Project | 2,119 | 4.4% |
| Administrative rights | 1,760 | 3.7% |

#### Stratified Splitting

```python
from sklearn.model_selection import train_test_split

# Stratified split preserves class proportions across splits
train, temp = train_test_split(df, test_size=0.3, stratify=df['Topic_group'])
val, test = train_test_split(temp, test_size=0.5, stratify=temp['Topic_group'])

# Result: Train (70%), Validation (15%), Test (15%)
```

**Output**: Dataset published to `saketgarodia1/IT-service-topic-classification-data`

---

### Part 2B: Teacher Model Training

**Notebook**: `Distillation_part_2.ipynb`

#### Model Architecture

```
BertForSequenceClassification
├── BertModel (109.5M params)
│   ├── Embeddings: word + position + token_type
│   ├── Encoder: 12 transformer layers
│   └── Pooler: [CLS] token projection
└── Classifier: Linear(768 → 8)
```

#### Training Configuration

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    lr_scheduler_type="linear",
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
)
```

#### Dynamic Padding

```python
# Instead of padding all sequences to max_length during preprocessing:
tokenizer(text, truncation=True, padding=False, max_length=256)

# Pad dynamically per-batch using DataCollatorWithPadding:
collator = DataCollatorWithPadding(tokenizer)
```

This approach is more memory-efficient as shorter batches use less memory.

#### Results

| Metric | Pre-Training | Post-Training |
|--------|--------------|---------------|
| Accuracy | 0.160 | 0.882 |
| F1 Macro | 0.072 | 0.879 |
| Eval Loss | - | 0.418 |

#### Per-Class Analysis

**Strongest Classes** (by recall):
- Hardware: 88%
- HR Support: 88%
- Access: 91%

**Strongest Classes** (by precision):
- Purchase: 95%
- Storage: 94%

**Output**: Model published to `saketgarodia1/bert-IT-ticket-classifier`

---

### Part 2C: Pooler-Only Training (Alternative Approach)

**Notebook**: `Distillation_part_2_train_pooler_only.ipynb`

This notebook demonstrates training only the classification head while keeping the BERT encoder frozen:

```python
# Freeze all BERT encoder layers
for name, param in model.bert.named_parameters():
    param.requires_grad = False

# Unfreeze pooler layer
for name, param in model.bert.named_parameters():
    if "pooler" in name:
        param.requires_grad = True

# Classifier is always trainable
for name, param in model.named_parameters():
    if "classifier" in name:
        param.requires_grad = True
```

**Trainable parameters**: ~590K (vs 109M for full fine-tuning)

This approach:
- Trains ~200x faster
- Uses less memory
- Works well when pre-trained representations are already strong
- Achieves ~52% accuracy (vs 88% for full fine-tuning) - demonstrating the value of full fine-tuning

---

### Part 2D: Post-Training Quantization

**Notebook**: `Distilliation_Part_IV.ipynb`

This notebook compares inference performance across different precision formats:

#### Quantization Comparison

| Format | Memory | Val Accuracy | Val F1 | Test Accuracy | Test F1 |
|--------|--------|--------------|--------|---------------|---------|
| FP32 | 255 MB | 0.8852 | 0.8848 | 0.8786 | 0.8755 |
| FP16 | 128 MB | 0.8852 | 0.8848 | 0.8785 | 0.8754 |
| INT4 (NF4) | 66 MB | 0.8843 | 0.8824 | 0.8799 | 0.8755 |

**Key Insight**: 4x memory reduction with <0.1% accuracy loss using NF4 quantization.

#### Why Some Parameters Stay in FP16

The notebook explains why embeddings, biases, and LayerNorm parameters are NOT quantized:

**Embeddings** (23.4M params):
- Represent semantic meaning of tokens
- Quantization destroys synonym relationships and semantic topology
- Causes 20-40% accuracy loss if quantized

**Biases** (768 params per layer):
- Control activation offsets
- Tiny memory footprint
- High sensitivity to rounding errors

**LayerNorm gamma/beta**:
- Critical for training stability
- Small errors compound across layers
- Cause gradient instability if quantized

#### 4-bit Configuration

```python
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # Double quantization for scales
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    quantization_config=nf4_config,
    device_map="auto",
)
```

**Output**: Model published to `saketgarodia1/distilbert-IT-ticket-student-4bit`

---

## Part 3: RLHF with Direct Preference Optimization (DPO)

**Notebook**: `RLHF_with_DPO.ipynb`

### What is DPO?

Direct Preference Optimization is an alternative to traditional RLHF that:

| Traditional RLHF (PPO) | DPO |
|------------------------|-----|
| Train reward model separately | No reward model needed |
| Complex RL optimization | Simple supervised loss |
| Unstable training | Stable, predictable |
| High compute requirements | Lower compute |

DPO directly optimizes the policy to prefer "chosen" responses over "rejected" ones using a closed-form loss function.

### Model & Dataset

| Component | Details |
|-----------|---------|
| Base Model | `teknium/OpenHermes-2.5-Mistral-7B` |
| Dataset | `Intel/orca_dpo_pairs` (12,859 preference pairs) |
| Format | {system, question, chosen, rejected} |

### Preference Pair Format

```json
{
  "system": "You are a helpful AI assistant.",
  "question": "What is the capital of France?",
  "chosen": "The capital of France is Paris, which is located in the north-central part of the country.",
  "rejected": "France's capital is Paris I think, or maybe Lyon?"
}
```

The model learns to generate responses more like "chosen" and less like "rejected".

### ChatML Format

```
<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris...<|im_end|>
```

### DPO Configuration

```python
DPOConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    warmup_steps=50,
    beta=0.1,                    # DPO temperature parameter
    max_prompt_length=1024,
    max_length=1536,
    bf16=True,
    gradient_checkpointing=True,
)
```

### The Beta Parameter

`beta` controls how strongly the model should deviate from the reference policy:

- **Low beta (0.1)**: Allows more deviation, stronger preference learning
- **High beta (0.5+)**: Keeps outputs closer to base model

### Training Process

```python
trainer = DPOTrainer(
    model=model,
    ref_model=None,           # Uses implicit reference
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

trainer.train()
# Final training loss: 0.040
```

### Output

- LoRA adapter merged with base model
- Published to: `saketgarodia1/OpenHermes-2.5-Mistral-7B-DPO`

---

## Published Models & Datasets

### Models on HuggingFace

| Model | Type | Base | Task |
|-------|------|------|------|
| [llama2-7b-guanaco-qlora-adapter](https://huggingface.co/saketgarodia1/llama2-7b-guanaco-qlora-adapter) | LoRA Adapter | Llama-2-7B | Instruction Following |
| [bert-IT-ticket-classifier](https://huggingface.co/saketgarodia1/bert-IT-ticket-classifier) | Full Model | BERT-base | IT Ticket Classification |
| [bert-it-ticket-student](https://huggingface.co/saketgarodia1/bert-it-ticket-student) | Distilled | DistilBERT | IT Ticket Classification |
| [distilbert-IT-ticket-student-4bit](https://huggingface.co/saketgarodia1/distilbert-IT-ticket-student-4bit) | Quantized | DistilBERT | IT Ticket Classification |
| [OpenHermes-2.5-Mistral-7B-DPO](https://huggingface.co/saketgarodia1/OpenHermes-2.5-Mistral-7B-DPO) | DPO-tuned | Mistral-7B | Chat/Instruction |

### Datasets on HuggingFace

| Dataset | Size | Task |
|---------|------|------|
| [IT-service-topic-classification-data](https://huggingface.co/datasets/saketgarodia1/IT-service-topic-classification-data) | 47.8k | Text Classification |
| [guanaco-llama2-chat-en](https://huggingface.co/datasets/saketgarodia1/guanaco-llama2-chat-en) | 3.7k | Instruction Tuning |

---

## Key Libraries & Concepts

### Core Libraries

| Library | Purpose | Key Features |
|---------|---------|--------------|
| `transformers` | Model loading, training | AutoModel, Trainer, Pipeline |
| `peft` | Parameter-efficient fine-tuning | LoRA, QLoRA, AdaLoRA |
| `trl` | RL-based training | SFTTrainer, DPOTrainer |
| `bitsandbytes` | Quantization | 4-bit NF4, 8-bit INT8 |
| `accelerate` | Distributed training | Multi-GPU, mixed precision |
| `datasets` | Data loading | HuggingFace Hub integration |

### Important Concepts

#### Gradient Checkpointing
Trades compute for memory by recomputing activations during backward pass instead of storing them:
```python
model.gradient_checkpointing_enable()
```

#### Gradient Accumulation
Simulates larger batch sizes on limited hardware:
```python
# Effective batch = per_device_batch * accumulation_steps
# 1 * 8 = 8
```

#### Dynamic Padding
Pads sequences to the longest in each batch, not globally:
```python
DataCollatorWithPadding(tokenizer)
```

#### Mixed Precision Training
Uses FP16/BF16 for forward pass, FP32 for gradients:
```python
TrainingArguments(bf16=True)
```

---

## Recommended Learning Path

### Beginner Path
1. **Start with Distillation Part 1** - Understand data preparation
2. **Then Distillation Part 2** - Learn full fine-tuning basics
3. **Then Part IV** - Understand quantization

### Intermediate Path
4. **SFT Notebook** - Learn QLoRA and instruction tuning
5. **Pooler-only training** - Understand transfer learning tradeoffs

### Advanced Path
6. **DPO Notebook** - Master preference-based alignment
7. **Experiment** - Combine techniques (e.g., SFT then DPO)

---

## Conclusion

This repository demonstrates the complete modern LLM training stack:

1. **SFT** teaches models to follow instructions
2. **Distillation** makes models deployable
3. **DPO** aligns models with human preferences

Each technique serves a specific purpose, and they're often used in combination:

```
Pre-trained LLM
     ↓
  [SFT] → Instruction-following model
     ↓
  [DPO] → Preference-aligned model
     ↓
[Distillation + Quantization] → Deployable model
```

Happy training!
