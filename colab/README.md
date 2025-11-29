# Google Colab Training Scripts

This directory contains automated training scripts optimized for Google Colab environments.

## Quick Start Scripts

### colab_train_and_upload.py
**Complete 1.5B model training pipeline**
- Fastest option (20-30 minutes on A100)
- Automated data collection, training, and upload
- Suitable for T4 or A100 GPUs
- Best for: Quick iteration and testing

Usage:
```python
!python colab_train_and_upload.py
```

### colab_train_7b.py
**Advanced 7B model training pipeline**
- Supports both standard and reasoning variants
- Requires A100 GPU (40GB VRAM)
- Training time: 2-3 hours
- Best for: Production-quality models

Usage:
```python
# Standard 7B model
!python colab_train_7b.py

# Reasoning variant (edit MODEL_VARIANT in script first)
!python colab_train_7b.py
```

## Setup Scripts

### setup_colab.py
**Environment setup and validation**
- Mounts Google Drive
- Clones/updates repository
- Verifies dependencies
- Checks GPU availability

### COLAB_QUICKSTART.py
**Legacy quick start script**
- Basic setup helper
- Use newer scripts above instead

## Utilities

### check_cuda_compatibility.py
**CUDA environment diagnostics**
- Validates CUDA installation
- Checks GPU compatibility
- Memory analysis
- Useful for troubleshooting

### COLAB_CELLS.md
**Ready-to-use Colab notebook cells**
- Copy-paste cells for quick setup
- Includes all training variants
- Monitoring and troubleshooting cells
- See file for complete documentation

## Prerequisites

### Colab Secrets Required
Add these to your Colab Secrets (key icon in sidebar):
- `GITHUB_TOKEN` - For data collection
- `HF_TOKEN` - For model upload
- `STACKOVERFLOW_KEY` - Optional

### Recommended Runtime
- **1.5B models**: T4 or A100 GPU
- **7B models**: A100 GPU (40GB) required

## Workflow

### 1.5B Model (Fastest)
```python
# Cell 1: Clone and setup
!git clone https://github.com/sylvester-francis/slm-typescript-model.git
%cd slm-typescript-model

# Cell 2: Train and upload
!python colab/colab_train_and_upload.py
```

### 7B Model (Production Quality)
```python
# Cell 1: Clone and setup
!git clone https://github.com/sylvester-francis/slm-typescript-model.git
%cd slm-typescript-model

# Cell 2: Train and upload
!python colab/colab_train_7b.py
```

## Output

All scripts automatically:
1. Collect and preprocess data
2. Train the model with optimal settings
3. Evaluate model performance
4. Create model card
5. Upload to HuggingFace
6. Backup to Google Drive

## Troubleshooting

### Out of Memory
```python
# Reduce batch size (edit in script)
BATCH_SIZE = 1
GRAD_ACCUM = 32
```

### Training Stuck
- Check GPU with `!nvidia-smi`
- View logs with `!tail -f training.log`
- First step compilation takes 1-2 minutes (normal)

### Import Errors
```python
# Reinstall dependencies
!pip install --upgrade transformers peft trl accelerate
```

## File Descriptions

| File | Purpose | When to Use |
|------|---------|-------------|
| `colab_train_and_upload.py` | 1.5B automated training | Quick models, testing |
| `colab_train_7b.py` | 7B advanced training | Production models |
| `setup_colab.py` | Environment setup | Manual setup needed |
| `check_cuda_compatibility.py` | GPU diagnostics | Troubleshooting |
| `COLAB_CELLS.md` | Notebook cells | Copy-paste workflow |
| `COLAB_QUICKSTART.py` | Legacy setup | Use newer scripts |

## See Also

- [Main Documentation](../docs/COLAB.md) - Detailed Colab guide
- [7B Training Guide](../docs/TRAINING_7B.md) - Advanced training
- [Main README](../README.md) - Project overview
