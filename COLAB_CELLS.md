# Google Colab Quick Start Cells

Copy and paste these cells directly into Google Colab to start training.

---

## Cell 1: Setup and Update (Run This First)

```python
# ============================================================================
# TypeScript SLM - Auto Setup & Update
# ============================================================================

print(" TypeScript SLM - Quick Start")
print("="*70)

# Mount Google Drive
print("\n Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# Setup project directory
import os
from pathlib import Path

project_dir = Path('/content/drive/MyDrive/slm_code')

if not project_dir.exists():
print("\n Cloning repository...")
os.chdir('/content/drive/MyDrive')
!git clone https://github.com/sylvester-francis/slm-typescript-model.git slm_code
else:
print("\n Updating repository...")
os.chdir(project_dir)
!git pull origin main

os.chdir(project_dir)
print(f"[OK] Working directory: {os.getcwd()}")

# Check GPU
print("\n GPU Information:")
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Verify Colab Secrets
print("\n Checking Colab Secrets...")
try:
from google.colab import userdata
for secret in ['GITHUB_TOKEN', 'HF_TOKEN']:
try:
userdata.get(secret)
print(f" [OK] {secret}")
except:
print(f" [ERROR] {secret} - Add via icon in sidebar")
except:
print(" [WARNING] Add GITHUB_TOKEN and HF_TOKEN to Colab Secrets")

print("\n" + "="*70)
print("[OK] Setup complete! Run one of the training cells below:")
print("="*70)
```

---

## Cell 2A: Train 1.5B Model (Fastest - 20-30 min)

```python
# Train 1.5B TypeScript SLM
# Dataset: 2k samples | Time: 20-30 min on A100 | Output: typescript-slm-1.5b
!python -u colab_train_and_upload.py
```

---

## Cell 2B: Train 7B Standard Model (Best Quality - 2-3 hours)

```python
# Train 7B Standard TypeScript SLM
# Dataset: 5k samples | Time: 2-3 hours on A100 | Output: typescript-slm-7b
!python -u colab_train_7b.py
```

---

## Cell 2C: Train 7B Reasoning Model (Advanced - 2-3 hours)

```python
# Train 7B Reasoning TypeScript SLM
# Dataset: 5k samples | Time: 2-3 hours on A100 | Output: typescript-slm-7b-reasoning

# Enable reasoning variant
!sed -i 's/MODEL_VARIANT = "standard"/MODEL_VARIANT = "reasoning"/' colab_train_7b.py

# Start training
!python -u colab_train_7b.py
```

---

## Cell 3: Monitor Training (Run in separate cell while training)

```python
# Watch training progress in real-time
!tail -f training.log
```

---

## Cell 4: Check Training Status

```python
# Check if training is running
!ps aux | grep "python.*colab_train" | grep -v grep

# Check GPU usage
!nvidia-smi

# View last 20 lines of log
!tail -20 training.log
```

---

## Cell 5: After Training - Download Model

```python
# Package and download trained model
import os

# Find the trained model directory
model_dirs = !ls -d models/typescript-slm-* 2>/dev/null
if model_dirs:
model_dir = model_dirs[0]
print(f" Packaging {model_dir}...")

# Create tarball
!tar -czf typescript-slm-model.tar.gz {model_dir}

# Copy to Google Drive
!cp typescript-slm-model.tar.gz /content/drive/MyDrive/

print(f"[OK] Model saved to Google Drive/typescript-slm-model.tar.gz")
print(f" Model size: {!du -h typescript-slm-model.tar.gz | cut -f1}")
else:
print("[ERROR] No trained model found. Training may not have completed.")
```

---

## Troubleshooting Cells

### If Training Seems Stuck

```python
# Check if actually running (look for python process)
!ps aux | grep python | grep -v grep

# Check last log update time
!ls -lh training.log

# Force kill if needed
!pkill -f "python.*colab_train"
```

### If Out of Memory

```python
# Restart runtime: Runtime → Restart runtime
# Then reduce batch size in the script:

# For 7B models:
!sed -i 's/BATCH_SIZE = 2/BATCH_SIZE = 1/' colab_train_7b.py
!sed -i 's/GRAD_ACCUM = 16/GRAD_ACCUM = 32/' colab_train_7b.py
!sed -i 's/LORA_R = 64/LORA_R = 32/' colab_train_7b.py

# Re-run training
!python -u colab_train_7b.py
```

### Check Environment

```python
# Verify all dependencies
!python -u scripts/check_environment.py

# Check Python packages
!pip list | grep -E "transformers|peft|trl|torch"

# Check disk space
!df -h /content/drive/MyDrive
```

---

## Quick Reference

| Model | Command | Time (A100) | Output |
|-------|---------|-------------|---------|
| 1.5B | `!python -u colab_train_and_upload.py` | 20-30 min | `typescript-slm-1.5b` |
| 7B Standard | `!python -u colab_train_7b.py` | 2-3 hours | `typescript-slm-7b` |
| 7B Reasoning | Edit then `!python -u colab_train_7b.py` | 2-3 hours | `typescript-slm-7b-reasoning` |

**All models automatically upload to HuggingFace:** `sylvester-francis/{model-name}`

---

## Tips

1. **Keep browser tab open** - Colab disconnects if browser is closed
2. **Monitor with Cell 3** - Run monitoring in separate cell
3. **Start with 1.5B** - Test setup before committing to 7B (2-3 hours)
4. **Check GPU** - Ensure you have A100 for 7B models (T4 insufficient)
5. **Backup to Drive** - Models auto-backup but run Cell 5 for manual download
