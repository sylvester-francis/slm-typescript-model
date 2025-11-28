# TypeScript SLM - Small Language Models for TypeScript

Fine-tuned small language models specialized in TypeScript code generation, optimized for React, Next.js, Angular, and Node.js frameworks.

## Overview

This project provides production-ready TypeScript code generation models built on Qwen 2.5 Coder with LoRA fine-tuning. The models are trained on high-quality, framework-specific TypeScript samples and optimized for modern web development workflows.

### Available Models

| Model | Size | Context | Best For | HuggingFace |
|-------|------|---------|----------|-------------|
| TypeScript SLM 1.5B | 1.5B params | 1024 tokens | Quick iteration, local development | [sylvester-francis/typescript-slm-1.5b](https://huggingface.co/sylvester-francis/typescript-slm-1.5b) |
| TypeScript SLM 7B | 7B params | 2048 tokens | Production code generation | [sylvester-francis/typescript-slm-7b](https://huggingface.co/sylvester-francis/typescript-slm-7b) |
| TypeScript SLM 7B Reasoning | 7B params | 2048 tokens | Complex problem-solving, debugging | [sylvester-francis/typescript-slm-7b-reasoning](https://huggingface.co/sylvester-francis/typescript-slm-7b-reasoning) |

## Quick Start

### Google Colab (Recommended)

Train your own model in 3 steps:

```python
# 1. Mount Drive and clone repository
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/
!git clone https://github.com/sylvester-francis/slm-typescript-model.git slm_code
%cd slm_code

# 2. Add tokens to Colab Secrets (click icon):
# - GITHUB_TOKEN
# - HF_TOKEN

# 3. Run training (20-30 min on A100)
!python colab_train_and_upload.py
```

See [Colab Guide](docs/COLAB.md) for detailed instructions.

### Local Installation

```bash
# Clone repository
git clone https://github.com/sylvester-francis/slm-typescript-model.git
cd slm-typescript-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-mac.txt # macOS
pip install -r requirements.txt # Linux/Colab

# Setup environment
cp .env.example .env
# Edit .env with your tokens
```

## Features

- **Framework-Optimized**: Specialized training on React, Next.js, Angular, and Node.js patterns
- **Quality-Scored Dataset**: Intelligent filtering prioritizes proper TypeScript usage and framework best practices
- **Multi-Platform**: Supports Google Colab (CUDA), Mac M-series (MPS), and Linux GPUs
- **Memory-Efficient**: LoRA fine-tuning enables training on consumer hardware
- **Production-Ready**: Automated pipeline from data collection to deployment

## Training Your Own Model

### 1.5B Model (Fastest)

**Hardware**: T4 GPU (16GB) or better
**Time**: 20-30 minutes on A100, 60-90 minutes on T4
**Use Case**: Quick iteration, testing, local development

```python
!python colab_train_and_upload.py
```

### 7B Standard Model (Best Quality)

**Hardware**: A100 GPU (40GB) required
**Time**: 2-3 hours
**Use Case**: Production code generation

```python
!python colab_train_7b.py
```

### 7B Reasoning Model (Advanced)

**Hardware**: A100 GPU (40GB) required
**Time**: 2-3 hours
**Use Case**: Complex problem-solving, debugging, architectural decisions

```python
# Edit colab_train_7b.py: MODEL_VARIANT = "reasoning"
!python colab_train_7b.py
```

## Model Performance

### Code Quality Comparison

Tested on 100 TypeScript generation tasks:

| Metric | 1.5B | 7B Standard | 7B Reasoning |
|--------|------|-------------|--------------|
| Correct Syntax | 85% | 94% | 95% |
| Proper TypeScript Types | 72% | 91% | 93% |
| Framework Best Practices | 68% | 89% | 92% |
| Context Understanding | 1024 tokens | 2048 tokens | 2048 tokens |

### Training Data Distribution

- **React** (43-58%): Components, hooks, context patterns
- **Angular** (33-50%): Services, directives, dependency injection
- **Next.js** (21-23%): Pages, API routes, SSR/SSG patterns
- **TypeScript** (9-16%): Advanced types, generics, utility types
- **Node.js** (6-11%): Express, NestJS, API servers

Quality scoring prioritizes:
- Proper type annotations (minimal `any` usage)
- Complete modules with imports/exports
- Framework-specific patterns
- Production-quality code from popular repositories

## Usage

### Basic Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
base_model,
device_map="auto",
torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load trained adapter
model = PeftModel.from_pretrained(
model,
"sylvester-francis/typescript-slm-1.5b"
)

# Generate code
prompt = "Create a React component with TypeScript for a user profile card:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Command Line Interface

```bash
# Complete pipeline
python cli.py pipeline

# Individual steps
python cli.py collect # Collect training data
python cli.py preprocess # Preprocess data
python cli.py train # Train model
python cli.py evaluate # Evaluate model
python cli.py upload # Upload to HuggingFace

# Custom training
python cli.py train \
--data data/processed/train_medium.jsonl \
--batch-size 4 \
--grad-accum 8 \
--lora-r 32 \
--epochs 3
```

## Project Structure

```
slm-typescript-model/
README.md # This file
docs/ # Documentation
COLAB.md # Colab training guide
TRAINING_7B.md # 7B models guide
TRAINING_MAC.md # Mac M-series training
MODEL_CARD.md # HuggingFace model card
cli.py # Main CLI interface
colab_train_and_upload.py # 1.5B automated training
colab_train_7b.py # 7B automated training
scripts/ # Core functionality
training.py # Training logic
filter_dataset.py # Dataset quality filtering
data_collection.py # GitHub/SO data collection
data_preprocessing.py # Data cleaning
evaluation.py # Model evaluation
upload_to_hf.py # HuggingFace upload
data/
raw/ # Raw collected data
processed/ # Filtered training datasets
train_small.jsonl # 2k samples (quick iteration)
train_ultra.jsonl # 3k samples (balanced)
train_medium.jsonl # 5k samples (recommended)
train.jsonl # 8k samples (maximum quality)
models/ # Trained models
```

## Documentation

- **[Colab Training Guide](docs/COLAB.md)** - Complete Google Colab setup and training
- **[7B Models Guide](docs/TRAINING_7B.md)** - Training larger models with reasoning
- **[Mac Training Guide](docs/TRAINING_MAC.md)** - Local training on Mac M-series
- **[Model Card](docs/MODEL_CARD.md)** - Detailed model specifications

## Requirements

### Minimum (1.5B Model)

- Python 3.10+
- 16GB RAM (local) or T4 GPU (Colab)
- 10GB disk space

### Recommended (7B Models)

- A100 GPU (40GB VRAM)
- 24GB RAM
- 40GB disk space

### Dependencies

- PyTorch 2.0+
- Transformers 4.36+
- PEFT 0.7+
- TRL 0.7+
- Accelerate 0.25+

## Environment Setup

Create a `.env` file in the project root:

```bash
# GitHub (required for data collection)
GITHUB_TOKEN=your_github_token

# HuggingFace (required for model upload)
HF_TOKEN=your_hf_token

# StackOverflow (optional)
STACKOVERFLOW_KEY=your_so_key
```

Get tokens:
- GitHub: https://github.com/settings/tokens (needs `repo` scope)
- HuggingFace: https://huggingface.co/settings/tokens (needs `write` access)
- StackOverflow: https://stackapps.com/apps/oauth/register

## Hardware Benchmarks

### Google Colab A100 (40GB)

| Model | Dataset | Training Time | Memory Usage |
|-------|---------|---------------|--------------|
| 1.5B | 2k samples | 20-30 min | ~20GB |
| 1.5B | 5k samples | 50-75 min | ~20GB |
| 7B | 5k samples | 2-3 hours | ~36GB |

### Google Colab T4 (16GB)

| Model | Dataset | Training Time | Memory Usage |
|-------|---------|---------------|--------------|
| 1.5B | 2k samples | 60-90 min | ~14GB |
| 1.5B | 5k samples | 150-180 min | ~14GB |
| 7B | Any | Not supported | Insufficient VRAM |

### Mac M4 (24GB Unified Memory)

| Model | Dataset | Training Time | Notes |
|-------|---------|---------------|-------|
| 1.5B | 2k samples | 10-12 hours | Works but slow |
| 1.5B | 5k+ samples | OOM | Insufficient memory |
| 7B | Any | Not supported | Insufficient memory |

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size and increase gradient accumulation
python cli.py train --batch-size 2 --grad-accum 16 --lora-r 16
```

**Mac M4 Out of Memory**
```bash
# Use smallest dataset only
python cli.py train --data data/processed/train_small.jsonl
```

**Training Stuck on First Step**
- Normal behavior - GPU compiling kernels
- Wait 1-2 minutes, training will accelerate

**Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Clear Python cache
rm -rf scripts/__pycache__
```

See [Colab Guide](docs/COLAB.md#troubleshooting) for detailed troubleshooting.

## Contributing

Contributions are welcome! Please:

1. Test changes on both Mac and Colab
2. Update relevant documentation
3. Follow existing code style
4. Add tests for new features

## Citation

If you use these models in your research or project, please cite:

```bibtex
@software{typescript_slm_2025,
author = {Francis, Sylvester},
title = {TypeScript SLM: Fine-tuned Small Language Models for TypeScript},
year = {2025},
url = {https://github.com/sylvester-francis/slm-typescript-model}
}
```

## Acknowledgments

- Built on [Qwen 2.5 Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) by Alibaba Cloud
- [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) for reasoning capabilities
- Training powered by [Hugging Face TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft)

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Links

- **Repository**: https://github.com/sylvester-francis/slm-typescript-model
- **Models**:
- [typescript-slm-1.5b](https://huggingface.co/sylvester-francis/typescript-slm-1.5b)
- [typescript-slm-7b](https://huggingface.co/sylvester-francis/typescript-slm-7b)
- [typescript-slm-7b-reasoning](https://huggingface.co/sylvester-francis/typescript-slm-7b-reasoning)
- **Issues**: https://github.com/sylvester-francis/slm-typescript-model/issues
