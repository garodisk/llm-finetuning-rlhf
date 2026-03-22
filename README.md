# LLM Training Techniques

A hands-on collection of Jupyter notebooks demonstrating three essential techniques for training and optimizing Large Language Models: **Supervised Fine-Tuning (SFT)**, **Knowledge Distillation**, and **RLHF via Direct Preference Optimization (DPO)**.

## Features

- **QLoRA Fine-Tuning**: Train Llama-2-7B on a single GPU using 4-bit quantization + LoRA
- **Knowledge Distillation**: Compress BERT to DistilBERT with minimal accuracy loss
- **Post-Training Quantization**: Compare FP32, FP16, and INT4 inference
- **DPO Training**: Align model outputs with human preferences without reward modeling
- **LLM-as-Judge Evaluation**: Evaluate model quality using GPT-4

## Quick Start

### Installation

```bash
pip install transformers>=4.57 datasets>=4.0 peft>=0.18 trl>=0.26
pip install accelerate bitsandbytes evaluate scikit-learn
pip install seaborn matplotlib tensorboard wandb
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8
- GPU with 16GB+ VRAM (T4, V100, A100, or RTX 3090/4090)

## Notebooks

| Notebook | Description | Model | Key Output |
|----------|-------------|-------|------------|
| `Supervised_Fine_Tuning_(SFT)_Finetuning.ipynb` | QLoRA instruction tuning | Llama-2-7B | 71% perplexity reduction |
| `Distillation_part_1.ipynb` | IT ticket dataset preparation | - | Stratified train/val/test splits |
| `Distillation_part_2.ipynb` | BERT classifier training | BERT-base | 88% accuracy, 88% F1 |
| `Distillation_part_2_train_pooler_only.ipynb` | Frozen encoder training | BERT-base | Pooler-only fine-tuning |
| `Distilliation_Part_IV.ipynb` | Quantization comparison | DistilBERT | FP32 vs FP16 vs INT4 |
| `RLHF_with_DPO.ipynb` | Direct Preference Optimization | Mistral-7B | Preference-aligned model |

## Results Summary

### SFT (Llama-2-7B)
| Metric | Before | After |
|--------|--------|-------|
| Perplexity | 11.24 | 3.28 |
| Eval Loss | 2.42 | 1.19 |

### Distillation (BERT Classification)
| Metric | Value |
|--------|-------|
| Test Accuracy | 88.2% |
| Test F1 (Macro) | 87.9% |

### Quantization Comparison
| Format | Memory | Accuracy |
|--------|--------|----------|
| FP32 | 255 MB | 88.52% |
| FP16 | 128 MB | 88.52% |
| INT4 | 66 MB | 88.43% |

## Models on HuggingFace

| Model | Link |
|-------|------|
| Llama-2 QLoRA Adapter | [saketgarodia1/llama2-7b-guanaco-qlora-adapter](https://huggingface.co/saketgarodia1/llama2-7b-guanaco-qlora-adapter) |
| BERT IT Classifier | [saketgarodia1/bert-IT-ticket-classifier](https://huggingface.co/saketgarodia1/bert-IT-ticket-classifier) |
| DistilBERT Student | [saketgarodia1/bert-it-ticket-student](https://huggingface.co/saketgarodia1/bert-it-ticket-student) |
| DistilBERT 4-bit | [saketgarodia1/distilbert-IT-ticket-student-4bit](https://huggingface.co/saketgarodia1/distilbert-IT-ticket-student-4bit) |
| Mistral DPO | [saketgarodia1/OpenHermes-2.5-Mistral-7B-DPO](https://huggingface.co/saketgarodia1/OpenHermes-2.5-Mistral-7B-DPO) |

## Datasets

| Dataset | Size | Link |
|---------|------|------|
| IT Service Tickets | 47.8k | [saketgarodia1/IT-service-topic-classification-data](https://huggingface.co/datasets/saketgarodia1/IT-service-topic-classification-data) |
| Guanaco Chat (English) | 3.7k | [saketgarodia1/guanaco-llama2-chat-en](https://huggingface.co/datasets/saketgarodia1/guanaco-llama2-chat-en) |

## Documentation

See [guide.md](guide.md) for a detailed technical walkthrough of each notebook, including:
- Step-by-step explanations of each technique
- Hyperparameter choices and their effects
- Code snippets with commentary
- Best practices and common pitfalls

## Project Structure

```
.
├── README.md
├── guide.md
├── Supervised_Fine_Tuning_(SFT)_Finetuning.ipynb
├── Distillation_part_1.ipynb
├── Distillation_part_2.ipynb
├── Distillation_part_2_train_pooler_only.ipynb
├── Distilliation_Part_IV.ipynb
└── RLHF_with_DPO.ipynb
```

## Recommended Learning Path

1. **Beginner**: Start with `Distillation_part_1.ipynb` and `Distillation_part_2.ipynb`
2. **Intermediate**: Move to `Supervised_Fine_Tuning_(SFT)_Finetuning.ipynb`
3. **Advanced**: Explore `RLHF_with_DPO.ipynb`

## Key Techniques Covered

- **QLoRA**: 4-bit quantization + Low-Rank Adaptation
- **Gradient Checkpointing**: Memory-efficient training
- **Dynamic Padding**: Batch-level padding efficiency
- **NF4 Quantization**: NormalFloat4 for transformers
- **DPO Loss**: Direct preference optimization without reward models
- **ChatML Format**: Structured chat templates

## License

MIT

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformers, peft, and trl libraries
- [Meta AI](https://ai.meta.com/) for Llama-2
- [Mistral AI](https://mistral.ai/) for Mistral-7B
- [Intel](https://www.intel.com/) for the Orca DPO dataset
