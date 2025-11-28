# Google Colab - Quick Start Guide

Run your TypeScript SLM training in just a few steps!

## Prerequisites

1. Google account with Colab access
2. Your API tokens:
   - GitHub token: https://github.com/settings/tokens
   - Hugging Face token: https://huggingface.co/settings/tokens
   - StackOverflow key (optional): https://stackapps.com/apps/oauth/register

## Step 1: Open Google Colab

Go to: https://colab.research.google.com/

## Step 2: Enable GPU

1. Click **Runtime → Change runtime type**
2. Select **GPU** (A100 recommended, T4 works too)
3. Click **Save**

## Step 3: Add Your Tokens to Colab Secrets

1. Click the **key icon 🔑** in the left sidebar
2. Add these secrets:
   - Name: `GITHUB_TOKEN` → Value: `your_github_token`
   - Name: `HF_TOKEN` → Value: `your_huggingface_token`
   - Name: `STACKOVERFLOW_KEY` → Value: `your_stackoverflow_key` (optional)

## Step 4: Run the Automated Script

Copy and paste this into a Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/
!git clone https://github.com/sylvester-francis/slm-typescript-model.git slm_code
%cd slm_code

# Run the complete pipeline
!python colab_train_and_upload.py
```

That's it! The script will:
- ✓ Set up the environment
- ✓ Train the model (~20-30 minutes on A100)
- ✓ Evaluate the model
- ✓ Upload to Hugging Face as `sylvester-francis/typescript-slm-1.5b`
- ✓ Save backup to Google Drive

## Customization

Edit `colab_train_and_upload.py` to customize:

```python
# Line 34-39: Configuration
DATASET = "data/processed/train_small.jsonl"  # Change to train_medium.jsonl or train.jsonl
BATCH_SIZE = 4        # Reduce to 2 if OOM
GRAD_ACCUM = 8        # Increase to 16 if reducing batch size
LORA_R = 32           # Reduce to 16 if OOM
EPOCHS = 3            # Number of training epochs
```

## Expected Training Times

| Dataset | Samples | A100 (40GB) | T4 (16GB) |
|---------|---------|-------------|-----------|
| train_small.jsonl | 2,000 | 20-30 min | 60-90 min |
| train_medium.jsonl | 5,000 | 50-75 min | 150-180 min |
| train.jsonl | 8,000 | 2-3 hours | 4-5 hours |

## Troubleshooting

**OOM Error:**
Reduce batch size in the script:
```python
BATCH_SIZE = 2
GRAD_ACCUM = 16
LORA_R = 16
```

**Script Fails:**
Check Colab Secrets are correctly added (click 🔑 icon to verify)

## After Training

Your model will be available at:
- **Hugging Face:** https://huggingface.co/sylvester-francis/typescript-slm-1.5b
- **Google Drive:** `MyDrive/typescript-slm-model.tar.gz`
- **Colab Directory:** `/content/drive/MyDrive/slm_code/models/typescript-slm-1.5b/`

## Manual Training (Alternative)

If you prefer step-by-step control, see [COLAB_GUIDE.md](COLAB_GUIDE.md)
