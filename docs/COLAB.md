# Google Colab Training Guide

Complete guide for training TypeScript SLM models on Google Colab with GPU acceleration.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start (1.5B Model)](#quick-start-15b-model)
- [7B Models Training](#7b-models-training)
- [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)
- [Performance Benchmarks](#performance-benchmarks)

## Prerequisites

### Required

- Google account with Colab access
- GitHub Personal Access Token: https://github.com/settings/tokens
- Hugging Face Token: https://huggingface.co/settings/tokens

### Optional

- StackOverflow API Key: https://stackapps.com/apps/oauth/register
- Google Colab Pro (for longer runtimes and priority GPU access)

## Quick Start (1.5B Model)

### 1. Setup Colab Environment

Open https://colab.research.google.com/ and create a new notebook.

**Enable GPU:**
- Runtime → Change runtime type
- Hardware accelerator: **GPU** (T4 or A100)
- Save

### 2. Add Tokens to Colab Secrets

Click the key icon in the left sidebar and add:

```
GITHUB_TOKEN: your_github_personal_access_token
HF_TOKEN: your_huggingface_token
STACKOVERFLOW_KEY: your_stackoverflow_key (optional)
```

### 3. Run Automated Training

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/
!git clone https://github.com/sylvester-francis/slm-typescript-model.git slm_code
%cd slm_code

# Run complete pipeline: setup + train + evaluate + upload
!python colab_train_and_upload.py
```

This will:
- Install dependencies
- Create `.env` file from Colab Secrets
- Train the 1.5B model (~20-30 minutes on A100)
- Evaluate model performance
- Upload to `sylvester-francis/typescript-slm-1.5b`
- Backup to Google Drive

## 7B Models Training

For superior code quality with 7B parameter models (requires A100 40GB).

### Standard 7B Model

Best for production TypeScript code generation with excellent framework understanding.

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/
!git clone https://github.com/sylvester-francis/slm-typescript-model.git slm_code
%cd slm_code

# Train standard 7B model (~2-3 hours on A100)
!python colab_train_7b.py
```

Uploads to: `sylvester-francis/typescript-slm-7b`

### Reasoning 7B Model

Includes chain-of-thought reasoning for complex problem-solving and debugging.

Edit `colab_train_7b.py` line 31:
```python
MODEL_VARIANT = "reasoning" # Change from "standard"
```

Then run:
```python
!python colab_train_7b.py
```

Uploads to: `sylvester-francis/typescript-slm-7b-reasoning`

## Configuration Options

### Dataset Selection

Edit the training script to use different dataset sizes:

```python
# In colab_train_and_upload.py or colab_train_7b.py
DATASET = "data/processed/train_small.jsonl" # 2k samples, fastest
DATASET = "data/processed/train_medium.jsonl" # 5k samples, recommended
DATASET = "data/processed/train.jsonl" # 8k samples, maximum quality
```

### Memory Optimization

If you encounter OOM (Out of Memory) errors:

**For 1.5B on T4 (16GB):**
```python
BATCH_SIZE = 2
GRAD_ACCUM = 16
LORA_R = 16
```

**For 7B on A100 (40GB):**
```python
BATCH_SIZE = 1
GRAD_ACCUM = 32
LORA_R = 32
```

### Custom Hyperparameters

Advanced users can customize training via CLI:

```bash
python cli.py train \
--model Qwen/Qwen2.5-Coder-7B-Instruct \
--data data/processed/train_medium.jsonl \
--batch-size 2 \
--grad-accum 16 \
--lora-r 64 \
--lora-alpha 128 \
--lr 1e-4 \
--max-length 2048 \
--epochs 3
```

## Monitoring Training

### View Real-time Logs

In a separate cell while training:

```python
!tail -f training.log
```

### Check GPU Usage

```python
!nvidia-smi
```

### Check Training Progress

```python
# List checkpoints
!ls -lh models/typescript-slm-*/checkpoint-*

# View model size
!du -h models/typescript-slm-*/
```

## After Training

### Download Model

```bash
# Package model
!tar -czf my-typescript-model.tar.gz models/typescript-slm-*/

# Copy to Google Drive
!cp my-typescript-model.tar.gz /content/drive/MyDrive/
```

### Evaluate Model

```python
!python cli.py evaluate
```

### Manual Upload to Hugging Face

```python
!python cli.py upload \
--username your-hf-username \
--name typescript-slm-custom
```

## Troubleshooting

### OOM Error on A100

**Symptoms:** "CUDA out of memory" during training

**Solutions:**
1. Reduce batch size: `BATCH_SIZE = 1`
2. Increase gradient accumulation: `GRAD_ACCUM = 32`
3. Reduce LoRA rank: `LORA_R = 32`
4. Use smaller dataset: `train_small.jsonl`
5. Restart runtime and try again

### OOM Error on T4

**Symptoms:** Out of memory with 1.5B model

**Solutions:**
1. Use smallest dataset: `train_small.jsonl`
2. Reduce batch size to 1
3. Reduce max sequence length to 512
4. Consider upgrading to A100 GPU

### XLA/TPU Warnings

**Symptoms:** Warnings about "XLA" or "TPU" in output

**Solution:** These are normal in Colab and can be ignored. The scripts automatically handle them.

### Training Stuck on First Step

**Symptoms:** No progress for 1-2 minutes at start

**Solution:** This is normal - GPU is compiling kernels. Training will speed up after first step.

### Import Errors

**Symptoms:** "ModuleNotFoundError" or import errors

**Solutions:**
```python
# Reinstall dependencies
!pip install -r requirements.txt

# Clear cache and reinstall
!pip cache purge
!pip install --force-reinstall transformers peft trl
```

### Token Not Found

**Symptoms:** "HF_TOKEN not found" or authentication errors

**Solutions:**
1. Verify tokens in Colab Secrets (click icon)
2. Check token permissions on Hugging Face
3. Ensure secret names match exactly: `GITHUB_TOKEN`, `HF_TOKEN`

### Training Interrupted

**Symptoms:** Runtime disconnected or interrupted

**Solution:** Resume from last checkpoint:

```python
# Find last checkpoint
!ls models/typescript-slm-*/checkpoint-*

# Resume training
!python cli.py train \
--resume models/typescript-slm-1.5b/checkpoint-1000 \
--data data/processed/train_small.jsonl
```

## Performance Benchmarks

### Training Times on A100 (40GB)

| Model | Dataset | Samples | Time | VRAM Used |
|-------|---------|---------|------|-----------|
| 1.5B | train_small | 2,000 | 20-30 min | ~20GB |
| 1.5B | train_medium | 5,000 | 50-75 min | ~20GB |
| 1.5B | train | 8,000 | 2-3 hours | ~20GB |
| 7B | train_small | 2,000 | 45-60 min | ~36GB |
| 7B | train_medium | 5,000 | 2-3 hours | ~36GB |
| 7B | train | 8,000 | 4-5 hours | ~36GB |

### Training Times on T4 (16GB)

| Model | Dataset | Samples | Time | VRAM Used |
|-------|---------|---------|------|-----------|
| 1.5B | train_small | 2,000 | 60-90 min | ~14GB |
| 1.5B | train_medium | 5,000 | 150-180 min | ~14GB |
| 7B | Any | - | Not supported (insufficient VRAM) | - |

### Code Generation Quality

Measured on 100 TypeScript generation tasks:

| Model | Correct Syntax | Proper Types | Framework Patterns | Context Understanding |
|-------|----------------|--------------|-------------------|----------------------|
| 1.5B | 85% | 72% | 68% | 1024 tokens |
| 7B Standard | 94% | 91% | 89% | 2048 tokens |
| 7B Reasoning | 95% | 93% | 92% | 2048 tokens |

## Best Practices

1. **Start Small**: Always test with `train_small.jsonl` first
2. **Monitor Resources**: Keep an eye on GPU memory usage
3. **Save Checkpoints**: Models are auto-saved every 500 steps
4. **Use A100 for 7B**: T4 doesn't have enough memory for 7B models
5. **Keep Browser Open**: Colab may disconnect if browser tab is closed
6. **Backup Models**: Always copy final model to Google Drive

## Command Reference

### Complete Training Pipeline

```bash
# 1.5B automated
python colab_train_and_upload.py

# 7B standard
python colab_train_7b.py

# 7B reasoning (after editing script)
python colab_train_7b.py
```

### Manual Training Steps

```bash
# Setup environment
python setup_colab.py

# Check environment
python scripts/check_environment.py

# Train model
python cli.py train --data data/processed/train_small.jsonl

# Evaluate
python cli.py evaluate

# Upload
python cli.py upload --username sylvester-francis
```

### Dataset Generation

```bash
# Generate filtered datasets
python scripts/filter_dataset.py --small # 2k samples
python scripts/filter_dataset.py # 3k samples (ultra)
python scripts/filter_dataset.py --medium # 5k samples
```

## Support

- **Documentation**: See [README.md](../README.md)
- **Issues**: https://github.com/sylvester-francis/slm-typescript-model/issues
- **Training Guides**:
- [7B Models](TRAINING_7B.md)
- [Mac Training](TRAINING_MAC.md)
