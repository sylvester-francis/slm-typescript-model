# TypeScript SLM - Small Language Model

A specialized small language model for TypeScript code generation and understanding, built on top of Qwen 2.5 Coder.

## Features

- 🚀 **Unified CLI** - Simple command-line interface for all operations
- 📊 **Data Collection** - Automated collection from GitHub and StackOverflow
- 🔧 **Data Preprocessing** - Clean and prepare training data
- 🎯 **Local Training** - Optimized for Mac M4 with 24GB RAM (also supports GPU/TPU)
- 📈 **Evaluation** - Test your model with various prompts
- ☁️ **Easy Upload** - Share your model on Hugging Face

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo>
cd slm&slr

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-mac.txt
```

### 2. Make CLI executable

```bash
chmod +x cli.py
```

### 3. Run the complete pipeline

```bash
# Option 1: Run everything at once
python cli.py pipeline

# Option 2: Run steps individually
python cli.py collect      # Collect data
python cli.py preprocess   # Clean and prepare data
python cli.py train        # Train the model
python cli.py evaluate     # Test the model
```

## CLI Commands

### System Information

```bash
python cli.py info
```

Shows system information including:
- Python and PyTorch versions
- Available device (MPS/CUDA/CPU)
- Training data status
- Trained models count

### Data Collection

```bash
python cli.py collect [OPTIONS]
```

**Options:**
- `--min-stars, -s` - Minimum GitHub stars (default: 1000)
- `--repo-limit, -r` - Repositories per framework (default: 5)
- `--so-limit` - StackOverflow questions (default: 50)

**Example:**
```bash
python cli.py collect --min-stars 2000 --repo-limit 10
```

### Data Preprocessing

```bash
python cli.py preprocess [OPTIONS]
```

**Options:**
- `--input, -i` - Input directory (default: data/raw)
- `--output, -o` - Output directory (default: data/processed)

**Example:**
```bash
python cli.py preprocess --input data/raw --output data/processed
```

### Training

```bash
python cli.py train [OPTIONS]
```

**Options:**
- `--model, -m` - Base model name (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)
- `--data, -d` - Training data path (default: data/processed/train.jsonl)
- `--output, -o` - Output directory (default: ./models/typescript-slm-1.5b)
- `--epochs, -e` - Number of epochs (default: 3)
- `--batch-size, -b` - Batch size (default: 4)
- `--grad-accum, -g` - Gradient accumulation steps (default: 4)
- `--lr` - Learning rate (default: 2e-4)
- `--max-length` - Max sequence length (default: 1024)
- `--lora-r` - LoRA rank (default: 64)
- `--save-steps` - Save checkpoint interval (default: 500)
- `--resume, -r` - Resume from checkpoint

**Examples:**

Basic training:
```bash
python cli.py train
```

Custom configuration:
```bash
python cli.py train \
  --epochs 5 \
  --batch-size 8 \
  --lr 1e-4 \
  --output ./models/my-model
```

Resume from checkpoint:
```bash
python cli.py train --resume ./models/typescript-slm-1.5b/checkpoint-1000
```

### Evaluation

```bash
python cli.py evaluate [OPTIONS]
```

**Options:**
- `--adapter, -a` - Path to adapter (default: ./models/typescript-slm-1.5b)
- `--model, -m` - Base model name (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)

**Example:**
```bash
python cli.py evaluate --adapter ./models/my-model
```

### Upload to Hugging Face

```bash
python cli.py upload [OPTIONS]
```

**Options:**
- `--model, -m` - Model directory path
- `--username, -u` - Hugging Face username (required)
- `--name, -n` - Model name on HF (default: typescript-slm-1.5b)

**Example:**
```bash
# Set your HF token first
export HF_TOKEN=your_token_here

python cli.py upload --username myusername --name my-ts-model
```

### Complete Pipeline

```bash
python cli.py pipeline [OPTIONS]
```

**Options:**
- `--collect/--no-collect` - Enable/disable data collection (default: enabled)
- `--preprocess/--no-preprocess` - Enable/disable preprocessing (default: enabled)
- `--train/--no-train` - Enable/disable training (default: enabled)
- `--evaluate/--no-evaluate` - Enable/disable evaluation (default: enabled)

**Examples:**

Run everything:
```bash
python cli.py pipeline
```

Skip data collection (use existing data):
```bash
python cli.py pipeline --no-collect
```

Only train and evaluate:
```bash
python cli.py pipeline --no-collect --no-preprocess
```

## Help

Get help for any command:

```bash
# General help
python cli.py --help

# Command-specific help
python cli.py train --help
python cli.py collect --help
```

## Environment Setup

### Required Environment Variables

Create a `.env` file in the project root:

```bash
# GitHub (required for data collection)
GITHUB_TOKEN=your_github_token

# StackOverflow (optional, increases rate limits)
STACKOVERFLOW_KEY=your_so_key

# Hugging Face (required for upload)
HF_TOKEN=your_hf_token
```

### Getting Tokens

**GitHub Token:**
1. Go to https://github.com/settings/tokens
2. Generate new token (classic)
3. Select `repo` scope
4. Copy token to `.env`

**Hugging Face Token:**
1. Go to https://huggingface.co/settings/tokens
2. Create new token
3. Select `write` access
4. Copy token to `.env`

## Hardware Requirements

### Minimum (CPU training - slow)
- 16GB RAM
- 50GB free disk space

### Recommended (Mac M4)
- Mac M4 with 24GB unified memory
- 50GB free disk space
- **Training time:** ~2-4 hours for 60K samples

### Optimal (Cloud TPU/GPU)
- Google Colab with TPU v2
- **Training time:** ~36-60 minutes for 60K samples

## Project Structure

```
slm&slr/
├── cli.py                  # Main CLI entry point
├── scripts/
│   ├── data_collection.py  # GitHub & StackOverflow collection
│   ├── data_preprocessing.py  # Data cleaning
│   ├── training.py         # Model training (formerly train_mac.py)
│   ├── evaluation.py       # Model evaluation
│   └── upload_to_hf.py    # Hugging Face upload
├── data/
│   ├── raw/               # Raw collected data
│   └── processed/         # Cleaned training data
├── models/                # Trained models
├── notebooks/             # Jupyter notebooks for Colab
├── requirements-mac.txt   # Python dependencies
└── .env                   # Environment variables (create this)
```

## Training Performance

| Hardware | Time (3 epochs, 60K samples) | Cost |
|----------|----------------------------|------|
| Mac M4 24GB | 2-4 hours | Free |
| Colab TPU v2 | 36-60 min | Free |
| Colab T4 GPU | 1.5-3 hours | Free |

## Troubleshooting

### "MPS backend out of memory"
Reduce batch size:
```bash
python cli.py train --batch-size 2 --grad-accum 8
```

### "No module named 'typer'"
Install dependencies:
```bash
pip install -r requirements-mac.txt
```

### "GITHUB_TOKEN not found"
Create `.env` file with your tokens (see Environment Setup above)

### Training is slow
Check device detection:
```bash
python cli.py info
```

Should show "Apple Metal (MPS)" for Mac M4.

## Tips

### 1. Test with small dataset first

```bash
# Create small test dataset
head -n 1000 data/processed/train.jsonl > data/processed/train_test.jsonl

# Quick test run
python cli.py train --data data/processed/train_test.jsonl --epochs 1
```

### 2. Monitor training

```bash
# In another terminal
tail -f training.log
```

### 3. Run overnight

```bash
# Run in background
nohup python cli.py train > output.log 2>&1 &

# Check progress
tail -f output.log
```

### 4. Save disk space

The CLI automatically keeps only the last 3 checkpoints to save space.

## Examples

### Complete workflow from scratch

```bash
# 1. Set up environment
echo "GITHUB_TOKEN=your_token" > .env
echo "HF_TOKEN=your_token" >> .env

# 2. Install dependencies
pip install -r requirements-mac.txt

# 3. Run everything
python cli.py pipeline

# 4. Upload to Hugging Face
python cli.py upload --username myusername
```

### Custom training configuration

```bash
# Train with specific settings
python cli.py train \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --data data/processed/train.jsonl \
  --output ./models/custom-model \
  --epochs 5 \
  --batch-size 6 \
  --grad-accum 3 \
  --lr 1e-4 \
  --max-length 2048
```

### Resume interrupted training

```bash
# Training was interrupted at step 1500
python cli.py train --resume ./models/typescript-slm-1.5b/checkpoint-1500
```

## Contributing

Feel free to open issues or submit PRs!

## License

MIT License - see LICENSE file for details
