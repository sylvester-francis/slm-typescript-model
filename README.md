# TypeScript SLM - Small Language Model

A specialized small language model for TypeScript code generation and understanding, optimized for React, Next.js, Angular, and Node.js frameworks. Built on Qwen 2.5 Coder 1.5B with LoRA fine-tuning.

## Features

- **Unified CLI** - Simple command-line interface for all operations
- **Intelligent Dataset Filtering** - Quality-scored TypeScript samples focused on popular frameworks
- **Multi-Platform Support** - Works on Mac M4 (MPS), Google Colab (CUDA), and local GPUs
- **Memory Optimized** - Efficient training for limited VRAM/RAM environments
- **Cross-Platform Compatible** - Automatic device detection and configuration

## Quick Start

### Prerequisites

- Python 3.10+
- 16GB RAM minimum (24GB recommended for Mac)
- 40GB VRAM for Google Colab A100 (or 10GB+ for other GPUs)

### Installation

```bash
# Clone the repository
git clone https://github.com/sylvester-francis/slm-typescript-model.git
cd slm-typescript-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-mac.txt  # or requirements.txt for Colab/Linux
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# GitHub (required for data collection)
GITHUB_TOKEN=your_github_token

# StackOverflow (optional)
STACKOVERFLOW_KEY=your_so_key

# Hugging Face (required for model upload)
HF_TOKEN=your_hf_token
```

## Training Options

### Dataset Sizes

The repository includes intelligent filtering to create multiple dataset tiers:

| Dataset | Samples | Size | Quality Score | Best For |
|---------|---------|------|---------------|----------|
| train_small.jsonl | 2,000 | 7.4MB | 47-64 | Quick iteration, testing |
| train_ultra.jsonl | 3,000 | 11MB | 46-64 | Balanced quality/speed |
| train_medium.jsonl | 5,000 | 17MB | 44-64 | High quality training |
| train.jsonl | 8,000 | 27MB | 41-64 | Maximum data |

### Generate Filtered Datasets

```bash
# Generate all dataset sizes
python scripts/filter_dataset.py --small   # 2k samples
python scripts/filter_dataset.py          # 3k samples (default)
python scripts/filter_dataset.py --medium # 5k samples
```

## Training on Google Colab (Recommended)

### Option 1: One-Command Automated Training (Easiest)

**First, add your tokens to Colab Secrets:**
1. Click the key icon 🔑 in the left sidebar of Colab
2. Add these secrets:
   - `GITHUB_TOKEN`: Your GitHub personal access token
   - `HF_TOKEN`: Your Hugging Face token
   - `STACKOVERFLOW_KEY`: Your StackOverflow key (optional)

**Then run the automated pipeline:**

```python
# In Colab notebook
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/
!git clone https://github.com/sylvester-francis/slm-typescript-model.git slm_code
%cd slm_code

# Run complete pipeline: setup + train + evaluate + upload
!python colab_train_and_upload.py
```

This automated script will:
- Mount Google Drive
- Clone/update repository
- Install dependencies
- Create .env file from Colab Secrets
- Train the model (A100: 20-30 min)
- Evaluate the model
- Upload to Hugging Face (username: sylvester-francis)
- Backup to Google Drive

**Note:** If .env file already exists in the repo, it will use that instead of creating a new one.

### Option 2: Manual Step-by-Step Setup

#### Initial Setup

```python
# In Colab notebook
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/
!git clone https://github.com/sylvester-francis/slm-typescript-model.git slm_code
%cd slm_code
!python setup_colab.py
```

### Training Commands

**For A100 40GB (Recommended settings):**

```bash
# Fast training (2k samples, 20-30 minutes)
python cli.py train \
  --data data/processed/train_small.jsonl \
  --batch-size 4 \
  --grad-accum 8 \
  --lora-r 32 \
  --epochs 3
```

**For T4 16GB:**

```bash
python cli.py train \
  --data data/processed/train_small.jsonl \
  --batch-size 2 \
  --grad-accum 16 \
  --lora-r 16 \
  --epochs 3
```

### Expected Training Times on Colab

| Dataset | A100 (40GB) | T4 (16GB) |
|---------|-------------|-----------|
| train_small.jsonl (2k) | 20-30 min | 60-90 min |
| train_ultra.jsonl (3k) | 30-45 min | 90-120 min |
| train_medium.jsonl (5k) | 50-75 min | 150-180 min |

## Training on Mac M4

**Note:** Mac M4 training is possible but has limitations due to memory constraints and slower MPS performance.

### Working Configuration

```bash
# Train with small dataset only (others may OOM)
python cli.py train \
  --data data/processed/train_small.jsonl \
  --batch-size 2 \
  --grad-accum 8 \
  --lora-r 16 \
  --epochs 3
```

**Expected time:** 10-12 hours for 2k samples

**Known issues:**
- Very slow first training step (5-10 minutes for MPS kernel compilation)
- High memory usage (may OOM with datasets larger than 2k samples)
- Recommended to use Colab instead for faster iteration

## CLI Commands

### Check Environment

```bash
# Verify setup and dependencies
python scripts/check_environment.py
```

### System Information

```bash
python cli.py info
```

Shows:
- Python and PyTorch versions
- Available device (MPS/CUDA/CPU)
- Training data status
- GPU/VRAM information

### Data Collection

```bash
python cli.py collect [OPTIONS]
```

Options:
- `--min-stars, -s` - Minimum GitHub stars (default: 1000)
- `--repo-limit, -r` - Repositories per framework (default: 5)
- `--so-limit` - StackOverflow questions (default: 50)

### Data Preprocessing

```bash
python cli.py preprocess [OPTIONS]
```

Options:
- `--input, -i` - Input directory (default: data/raw)
- `--output, -o` - Output directory (default: data/processed)

### Training

```bash
python cli.py train [OPTIONS]
```

Key Options:
- `--data, -d` - Training data path
- `--batch-size, -b` - Batch size (default: 4)
- `--grad-accum, -g` - Gradient accumulation (default: 8)
- `--lora-r` - LoRA rank (default: 32)
- `--epochs, -e` - Number of epochs (default: 3)
- `--max-samples` - Limit dataset size for testing
- `--resume, -r` - Resume from checkpoint

Examples:

```bash
# Basic training with defaults
python cli.py train --data data/processed/train_small.jsonl

# Custom configuration
python cli.py train \
  --data data/processed/train_medium.jsonl \
  --batch-size 8 \
  --grad-accum 4 \
  --lora-r 64 \
  --epochs 5

# Quick test with 1000 samples
python cli.py train \
  --data data/processed/train_small.jsonl \
  --max-samples 1000 \
  --epochs 1

# Resume from checkpoint
python cli.py train \
  --resume models/typescript-slm-1.5b/checkpoint-100
```

### Evaluation

```bash
python cli.py evaluate [OPTIONS]
```

Options:
- `--adapter, -a` - Path to trained adapter
- `--model, -m` - Base model name

### Upload to Hugging Face

```bash
# Set token first
export HF_TOKEN=your_token

python cli.py upload \
  --username your-username \
  --name typescript-slm-1.5b
```

### Complete Pipeline

```bash
python cli.py pipeline [OPTIONS]
```

The pipeline is smart and automatically skips steps if data already exists.

Options:
- `--collect/--no-collect` - Run data collection
- `--preprocess/--no-preprocess` - Run preprocessing
- `--train/--no-train` - Run training
- `--evaluate/--no-evaluate` - Run evaluation
- `--max-samples` - Limit dataset size
- `--force, -f` - Force re-download and re-process

Examples:

```bash
# Run everything (auto-skips existing data)
python cli.py pipeline

# Force re-download everything
python cli.py pipeline --force

# Only train with existing data
python cli.py pipeline --no-collect --no-preprocess --no-evaluate

# Quick test with limited samples
python cli.py pipeline --max-samples 1000 --no-evaluate
```

## Project Structure

```
slm-typescript-model/
├── cli.py                      # Main CLI entry point
├── setup_colab.py              # Colab setup automation
├── COLAB_GUIDE.md             # Detailed Colab instructions
├── scripts/
│   ├── __init__.py            # Package init
│   ├── training.py            # Training logic
│   ├── filter_dataset.py      # Dataset filtering
│   ├── data_collection.py     # GitHub & SO collection
│   ├── data_preprocessing.py  # Data cleaning
│   ├── evaluation.py          # Model evaluation
│   ├── check_environment.py   # Environment validation
│   └── upload_to_hf.py       # HF upload
├── data/
│   ├── raw/                   # Raw collected data
│   └── processed/             # Filtered training data
│       ├── train_small.jsonl  # 2k samples
│       ├── train_ultra.jsonl  # 3k samples
│       ├── train_medium.jsonl # 5k samples
│       └── train.jsonl        # 8k samples
├── models/                    # Trained models
└── .env                       # Environment variables

```

## Troubleshooting

### CUDA Out of Memory (Colab)

**Solution 1:** Reduce batch size and LoRA rank
```bash
python cli.py train \
  --data data/processed/train_small.jsonl \
  --batch-size 2 \
  --grad-accum 16 \
  --lora-r 16
```

**Solution 2:** Restart runtime to clear memory
- Runtime → Restart runtime
- Re-run setup cells

**Solution 3:** Use memory fragmentation fix
```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

### MPS Out of Memory (Mac M4)

**This is expected.** Mac M4 with 24GB RAM cannot handle datasets larger than 2k samples.

**Solutions:**
- Use train_small.jsonl only
- Reduce batch size to 1: `--batch-size 1 --grad-accum 16`
- Train on Colab instead (recommended)

### Training Stuck on First Step (Mac M4)

**This is normal.** The first step takes 5-10 minutes while MPS compiles kernels. Subsequent steps will be faster (but still slow at ~10-20 seconds each).

### Import Errors

```bash
# Missing dependencies
pip install -r requirements-mac.txt

# Clear Python cache
rm -rf scripts/__pycache__
```

### XLA/TPU Warnings (Colab)

These warnings are normal and can be ignored. The training script handles them automatically.

## Performance Benchmarks

### Google Colab A100 (40GB VRAM)

| Configuration | Time per Step | Total Time (2k samples, 3 epochs) |
|--------------|---------------|-----------------------------------|
| Batch 4, LoRA 32 | ~6-8 sec | 20-30 min |
| Batch 8, LoRA 64 | ~10-12 sec | 35-45 min |
| Batch 16, LoRA 128 | OOM | N/A |

### Mac M4 (24GB RAM)

| Configuration | Time per Step | Issues |
|--------------|---------------|--------|
| Batch 2, LoRA 16 | ~90 sec | Works with train_small.jsonl only |
| Batch 4, LoRA 32 | OOM | Out of memory |
| Batch 8+ | OOM | Out of memory |

## Model Quality

The filtered datasets focus on high-quality TypeScript code with proper type annotations, focusing on:

- **React** (43-58% of samples) - Components, hooks, context
- **Angular** (33-50% of samples) - Services, directives, modules
- **Next.js** (21-23% of samples) - Pages, API routes, SSR
- **TypeScript** (9-16% of samples) - Advanced types, generics
- **Node.js** (6-11% of samples) - Express, NestJS, APIs

Quality scoring prioritizes:
- Interface and type definitions
- Proper type annotations (not overusing `any`)
- Complete modules with imports/exports
- Framework-specific patterns
- Production-quality code from popular repositories

## Tips and Best Practices

### 1. Start Small

Always test with train_small.jsonl first:
```bash
python cli.py train \
  --data data/processed/train_small.jsonl \
  --epochs 1 \
  --max-samples 500
```

### 2. Monitor Training

```bash
# Real-time monitoring
tail -f training.log

# Check GPU usage (Colab)
watch -n 1 nvidia-smi
```

### 3. Save Checkpoints

The CLI automatically saves checkpoints every 500 steps. Keep the `models/` directory backed up.

### 4. Use Colab for Production

For final training runs, use Google Colab with A100:
- 10-20x faster than Mac M4
- No memory constraints
- Can train full 8k dataset

### 5. Iterate Quickly

Use the smallest dataset for experimentation, then scale up once confident:
1. Test with 500 samples, 1 epoch
2. Train with 2k samples, 3 epochs
3. Final training with 5k-8k samples, 3 epochs

## Getting Tokens

### GitHub Token
1. Go to https://github.com/settings/tokens
2. Generate new token (classic)
3. Select `repo` scope
4. Copy token to `.env`

### Hugging Face Token
1. Go to https://huggingface.co/settings/tokens
2. Create new token
3. Select `write` access
4. Copy token to `.env`

## Documentation

- [COLAB_GUIDE.md](COLAB_GUIDE.md) - Comprehensive Colab training guide
- [TRAINING_MAC.md](TRAINING_MAC.md) - Mac-specific training notes

## Contributing

Contributions welcome! Please:
1. Test changes on both Mac and Colab
2. Update relevant documentation
3. Follow existing code style
4. Add tests for new features

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built on [Qwen 2.5 Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- Uses [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning
- Training pipeline powered by [TRL](https://github.com/huggingface/trl)
