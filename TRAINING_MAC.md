# Training TypeScript SLM on Mac M4

This guide shows you how to train the TypeScript SLM model locally on your Mac M4 with 24GB RAM.

> **Note:** We now provide a unified CLI for all operations. See the [README.md](README.md) for complete CLI documentation.

## Prerequisites

- Mac M4 with 24GB unified memory
- macOS 13+ (for optimal MPS support)
- Python 3.9 or higher
- ~50GB free disk space (for model downloads and checkpoints)

## Setup

### 1. Create a Python virtual environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements-mac.txt
```

**Note:** PyTorch should automatically detect and use Apple's MPS (Metal Performance Shaders) for GPU acceleration.

### 3. Verify MPS availability

```bash
# Using the CLI
python cli.py info

# Or manually
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

You should see: `MPS available: True`

## Training

### Basic usage (default settings)

```bash
# Using the new CLI (recommended)
python cli.py train

# Or directly via the training script
python scripts/training.py
```

This will:
- Use the dataset at `data/processed/train.jsonl`
- Train for 3 epochs
- Save the model to `./models/typescript-slm-1.5b`
- Create logs in `training.log`

### Custom configuration

```bash
# Using the CLI
python cli.py train \
  --data data/processed/train.jsonl \
  --output ./models/my-model \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 2e-4 \
  --max-length 1024
```

> **CLI Benefits:** The new CLI provides better help text, auto-completion, and consistent command structure across all operations.

### Available options

Get full help for all options:

```bash
python cli.py train --help
```

Key options:
- `--model, -m` - Base model name (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)
- `--data, -d` - Training data path (default: data/processed/train.jsonl)
- `--output, -o` - Output directory (default: ./models/typescript-slm-1.5b)
- `--epochs, -e` - Number of epochs (default: 3)
- `--batch-size, -b` - Batch size (default: 4)
- `--grad-accum, -g` - Gradient accumulation (default: 4)
- `--lr` - Learning rate (default: 2e-4)
- `--max-length` - Max sequence length (default: 1024)
- `--lora-r` - LoRA rank (default: 64)
- `--save-steps` - Checkpoint interval (default: 500)
- `--resume, -r` - Resume from checkpoint (optional)

## Memory and Performance Tuning

### For 24GB Mac M4 (recommended):
- Batch size: 4
- Gradient accumulation: 4
- Max sequence length: 1024
- **Effective batch size:** 16

### If you run out of memory:
Reduce batch size and increase gradient accumulation:

```bash
python cli.py train --batch-size 2 --grad-accum 8
```

### To train faster (if stable):
Increase batch size:

```bash
python cli.py train --batch-size 6 --grad-accum 3
```

## Expected Training Time

For **58,933 samples** with default settings:
- **~2-4 hours** for 3 epochs on Mac M4
- Progress is logged every 10 steps
- Checkpoints saved every 500 steps

## Monitoring Progress

### View live training logs:

```bash
tail -f training.log
```

### Training output shows:
- Loss values (should decrease over time)
- Learning rate
- Steps/second
- Estimated time remaining

Example output:
```
{'loss': 2.1234, 'learning_rate': 0.0002, 'epoch': 0.5}
Step 500/11049 [00:45<02:30, 5.2it/s]
```

## Checkpoints

Checkpoints are automatically saved every 500 steps to `{output_dir}/checkpoint-{step}`.

### Resume from checkpoint:

```bash
python cli.py train --resume ./models/typescript-slm-1.5b/checkpoint-1000
```

### Interrupt training (Ctrl+C):

If you interrupt training with Ctrl+C, the current progress is saved to:
```
{output_dir}/interrupted_checkpoint
```

Resume it later:
```bash
python cli.py train --resume ./models/typescript-slm-1.5b/interrupted_checkpoint
```

## Output Files

After training completes, you'll find:

```
models/typescript-slm-1.5b/
├── adapter_config.json      # LoRA adapter configuration
├── adapter_model.safetensors # LoRA adapter weights
├── tokenizer_config.json    # Tokenizer configuration
├── tokenizer.json           # Tokenizer vocabulary
├── special_tokens_map.json  # Special tokens
├── logs/                    # Training logs
└── checkpoint-*/            # Saved checkpoints
```

## Tips

### 1. Run overnight
Training takes 2-4 hours, so you can start it before bed:

```bash
nohup python cli.py train > training_output.log 2>&1 &
```

### 2. Test with small dataset first
Create a small test dataset:

```bash
head -n 1000 data/processed/train.jsonl > data/processed/train_test.jsonl
python cli.py train --data data/processed/train_test.jsonl --epochs 1
```

This runs in ~5-10 minutes to verify everything works.

### 3. Monitor system resources
Open Activity Monitor to watch:
- Memory usage (should stay under 24GB)
- CPU/GPU usage
- Disk activity

### 4. Keep your Mac plugged in
Training is intensive - make sure you're connected to power.

## Troubleshooting

### "MPS backend out of memory"
Reduce batch size:
```bash
python cli.py train --batch-size 2
```

### "No module named 'typer'" or "No module named 'transformers'"
Activate your virtual environment:
```bash
source venv/bin/activate
pip install -r requirements-mac.txt
```

### Training is very slow
Check device detection:
```bash
python cli.py info
```

Should show "Apple Metal (MPS)" for Mac M4.

### Dataset not found
Make sure the path is correct:
```bash
ls -lh data/processed/train.jsonl
```

## Next Steps

After training completes, you can:

1. **Test the model** - Create an inference script
2. **Upload to Hugging Face** - Share your trained model
3. **Fine-tune further** - Run more epochs with lower learning rate
4. **Merge LoRA adapters** - Combine with base model for easier deployment

## Comparison: Mac M4 vs Colab

| Feature | Mac M4 24GB | Colab TPU v2 |
|---------|-------------|--------------|
| Training time | 2-4 hours | 36-60 min |
| Cost | Free (your hardware) | Free |
| Time limit | Unlimited | 12 hours |
| Resumable | Yes | No (session ends) |
| Internet required | Only for downloads | Always |
| Control | Full control | Limited |

**Recommendation:**
- For one-time training: Use Colab TPU (faster)
- For experimentation: Use Mac M4 (more convenient)
- For production: Use Mac M4 or cloud GPU
