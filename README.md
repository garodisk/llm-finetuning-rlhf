# LLM Fine-Tuning & Alignment

Hands-on Jupyter notebooks demonstrating how to **fine-tune** and **align** Large Language Models using two essential techniques: **Supervised Fine-Tuning (SFT)** with QLoRA and **RLHF via Direct Preference Optimization (DPO)**.

## What's Inside

| Technique | What It Does | Model | Dataset |
|-----------|--------------|-------|---------|
| **SFT** | Teaches the model to follow instructions | Llama-2-7B | Guanaco chat (3.7k examples) |
| **DPO** | Aligns outputs with human preferences | Mistral-7B | Orca preference pairs (12.8k) |

## Quick Start

### Installation

```bash
pip install transformers>=4.57 datasets>=4.0 peft>=0.18 trl>=0.26
pip install accelerate bitsandbytes evaluate
pip install tensorboard wandb
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8
- GPU with 16GB+ VRAM (T4, V100, A100, or RTX 3090/4090)

## Notebooks

| Notebook | Description | Key Technique |
|----------|-------------|---------------|
| `Supervised_Fine_Tuning_(SFT)_Finetuning.ipynb` | Fine-tune Llama-2-7B on instruction data | QLoRA (4-bit + LoRA) |
| `RLHF_with_DPO.ipynb` | Align Mistral-7B with preference data | Direct Preference Optimization |

## Results

### SFT (Llama-2-7B)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Perplexity | 11.24 | 3.28 | -71% |
| Eval Loss | 2.42 | 1.19 | -51% |

**Trainable Parameters**: 40M (0.59% of 6.8B) using QLoRA

### DPO (Mistral-7B)

- Trained on 12.8k preference pairs (chosen vs rejected responses)
- Final training loss: 0.040
- Model learns to generate "preferred" style responses

## Models on HuggingFace

| Model | Link |
|-------|------|
| Llama-2 QLoRA Adapter | [saketgarodia1/llama2-7b-guanaco-qlora-adapter](https://huggingface.co/saketgarodia1/llama2-7b-guanaco-qlora-adapter) |
| Mistral DPO | [saketgarodia1/OpenHermes-2.5-Mistral-7B-DPO](https://huggingface.co/saketgarodia1/OpenHermes-2.5-Mistral-7B-DPO) |

## The Training Pipeline

These two techniques are typically used in sequence:

```
Pre-trained LLM
      │
      ▼
   [ SFT ]  ──→  Model follows instructions
      │
      ▼
   [ DPO ]  ──→  Model outputs align with preferences
      │
      ▼
 Production-ready assistant
```

## Key Techniques Covered

- **QLoRA**: 4-bit quantization + Low-Rank Adaptation for memory-efficient training
- **LoRA Targets**: Adapting attention (q, k, v, o) and FFN (gate, up, down) projections
- **DPO Loss**: Preference learning without a separate reward model
- **ChatML Format**: Structured chat templates for instruction models
- **Gradient Checkpointing**: Trade compute for memory
- **LLM-as-Judge**: Using GPT-4 to evaluate model outputs

## Documentation

See [guide.md](guide.md) for detailed technical explanations including:
- Step-by-step notebook walkthroughs
- Hyperparameter choices and effects
- Code snippets with commentary

## Project Structure

```
.
├── README.md
├── guide.md
├── Supervised_Fine_Tuning_(SFT)_Finetuning.ipynb
└── RLHF_with_DPO.ipynb
```

## Suggested Repo Names

- `llm-sft-dpo` - Short and descriptive
- `finetune-align-llm` - Action-oriented
- `llm-instruction-alignment` - Describes the goal
- `qlora-dpo-tutorial` - Names the techniques

## License

MIT

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformers, peft, and trl
- [Meta AI](https://ai.meta.com/) for Llama-2
- [Mistral AI](https://mistral.ai/) for Mistral-7B
- [Intel](https://www.intel.com/) for the Orca DPO dataset
