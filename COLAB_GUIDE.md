# Google Colab Training Guide

Quick guide to train your TypeScript SLM on Google Colab with A100 GPU.

## Prerequisites

1. **Google Account** with Colab access
2. **GitHub repo** already set up (you have this!)
3. **Google Drive** with space for models (~2-3GB)

## Setup (One-time)

### 1. Open Google Colab

Go to: https://colab.research.google.com/

### 2. Create New Notebook

Click: **File → New Notebook**

### 3. Enable GPU

- Click: **Runtime → Change runtime type**
- Hardware accelerator: **GPU** (preferably A100 or T4)
- Click: **Save**

### 4. Run Setup

Copy and paste these commands into the first cell:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project directory
%cd /content/drive/MyDrive/

# Clone repository (first time only)
!git clone https://github.com/sylvester-francis/slm-typescript-model.git slm_code

# Or if already cloned, just update
%cd slm_code
!git pull origin main

# Run setup script
!python setup_colab.py
```

## Training Commands

### Quick Test (~10-15 minutes)
```python
!python cli.py train \
  --data data/processed/train_small.jsonl \
  --batch-size 32 \
  --grad-accum 1 \
  --lora-r 128 \
  --epochs 3
```

### Medium Quality (~25-35 minutes)
```python
!python cli.py train \
  --data data/processed/train_medium.jsonl \
  --batch-size 32 \
  --grad-accum 1 \
  --lora-r 128 \
  --epochs 3
```

### Full Dataset (~45-60 minutes)
```python
!python cli.py train \
  --data data/processed/train.jsonl \
  --batch-size 32 \
  --grad-accum 1 \
  --lora-r 128 \
  --epochs 3
```

## Monitor Training

### Check Progress
```python
# View training log in real-time
!tail -f training.log
```

### Check GPU Usage
```python
!nvidia-smi
```

### Check Model Output
```python
!ls -lh models/typescript-slm-1.5b/
```

## Generate Datasets (if needed)

If you don't have the filtered datasets, create them:

```python
# Generate all three sizes
!python scripts/filter_dataset.py --small   # 2k samples
!python scripts/filter_dataset.py          # 3k samples (ultra)
!python scripts/filter_dataset.py --medium # 5k samples
```

## Download Trained Model

After training completes:

```python
# Compress model for download
!tar -czf typescript-slm-model.tar.gz models/typescript-slm-1.5b/

# Download via Colab files panel (left sidebar)
# Or copy to Google Drive
!cp typescript-slm-model.tar.gz /content/drive/MyDrive/
```

## Troubleshooting

### "Out of Memory" Error

Reduce batch size:
```python
!python cli.py train \
  --data data/processed/train_small.jsonl \
  --batch-size 16 \
  --grad-accum 2 \
  --lora-r 64
```

### "XLA/TPU" Warnings

These are normal in Colab and can be ignored. The script handles them automatically.

### Training Interrupted

Resume from checkpoint:
```python
!python cli.py train \
  --data data/processed/train_small.jsonl \
  --resume models/typescript-slm-1.5b/checkpoint-XXX \
  --batch-size 32
```

## Performance Tips

1. **Use A100 GPU** if available (fastest)
2. **Start with `train_small.jsonl`** to verify everything works
3. **Keep browser tab open** - Colab may disconnect if idle
4. **Use Colab Pro** for longer runtime (optional, $10/month)

## Expected Training Times

| Dataset | Samples | A100 Time | T4 Time |
|---------|---------|-----------|---------|
| train_small.jsonl | 2,000 | ~10-15 min | ~30-40 min |
| train_ultra.jsonl | 3,000 | ~15-20 min | ~45-60 min |
| train_medium.jsonl | 5,000 | ~25-35 min | ~75-90 min |
| train.jsonl | 8,000 | ~45-60 min | ~2-3 hours |

## Complete Colab Notebook Template

```python
# ==============================================================================
# TypeScript SLM Training on Google Colab
# ==============================================================================

# 1. Setup
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/slm_code
!git pull origin main
!python setup_colab.py

# 2. Check Environment
!python scripts/check_environment.py

# 3. Train (choose one)
# Fast test:
!python cli.py train --data data/processed/train_small.jsonl --batch-size 32 --grad-accum 1 --lora-r 128

# OR Medium quality:
# !python cli.py train --data data/processed/train_medium.jsonl --batch-size 32 --grad-accum 1 --lora-r 128

# 4. Monitor (run in separate cell while training)
!tail -f training.log

# 5. Evaluate (after training)
!python cli.py evaluate

# 6. Package model for download
!tar -czf typescript-slm-model.tar.gz models/typescript-slm-1.5b/
!cp typescript-slm-model.tar.gz /content/drive/MyDrive/
print("✅ Model saved to Google Drive!")
```

## Next Steps

After training completes:
1. Download the model
2. Test it locally with `python cli.py evaluate`
3. Upload to Hugging Face (optional): `python cli.py upload`
4. Use in your coding agent!

## Support

For issues, check:
- Training log: `training.log`
- GitHub Issues: https://github.com/sylvester-francis/slm-typescript-model/issues
