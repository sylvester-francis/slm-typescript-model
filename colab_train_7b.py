#!/usr/bin/env python3
"""
Advanced Colab Training Script for 7B TypeScript SLM with Reasoning
Supports both standard and reasoning model variants
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description, check=True):
"""Run a shell command and print status"""
print(f"\n{'='*70}")
print(f" {description}")
print(f"{'='*70}")
print(f"Running: {cmd}\n")

result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

if check and result.returncode != 0:
print(f"\n[ERROR] Failed: {description}")
sys.exit(1)

print(f"\n[OK] Success: {description}")
return result.returncode == 0

def main():
print("\n" + "="*70)
print(" TypeScript SLM 7B - Advanced Training Pipeline")
print("="*70)

# ========== CONFIGURATION ==========
# Model Selection - Choose one:
MODEL_VARIANT = "standard" # Options: "standard" or "reasoning"

# Model configurations
MODELS = {
"standard": {
"name": "Qwen/Qwen2.5-Coder-7B-Instruct",
"hf_repo": "typescript-slm-7b",
"description": "Standard 7B TypeScript model"
},
"reasoning": {
"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
"hf_repo": "typescript-slm-7b-reasoning",
"description": "7B TypeScript model with reasoning capabilities"
}
}

# Training Configuration
HF_USERNAME = "sylvester-francis"
DATASET = "data/processed/train_medium.jsonl" # Use medium (5k) for 7B

# Memory-optimized settings for A100 40GB
BATCH_SIZE = 2 # Smaller batch for 7B model
GRAD_ACCUM = 16 # Higher accumulation for effective batch 32
LORA_R = 64 # Higher rank for 7B model
LORA_ALPHA = 128 # 2x rank
EPOCHS = 3
MAX_SEQ_LENGTH = 2048 # Longer context for 7B
LEARNING_RATE = 1e-4 # Lower LR for larger model

# Get selected model config
model_config = MODELS[MODEL_VARIANT]
MODEL_NAME = model_config["name"]
HF_REPO_NAME = model_config["hf_repo"]
OUTPUT_DIR = f"./models/{HF_REPO_NAME}"

print(f"\nConfiguration:")
print(f" Model Variant: {MODEL_VARIANT.upper()}")
print(f" Base Model: {MODEL_NAME}")
print(f" Description: {model_config['description']}")
print(f" HF Username: {HF_USERNAME}")
print(f" HF Repo: {HF_REPO_NAME}")
print(f" Dataset: {DATASET}")
print(f" Batch Size: {BATCH_SIZE}")
print(f" Gradient Accumulation: {GRAD_ACCUM}")
print(f" Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM}")
print(f" LoRA Rank: {LORA_R}")
print(f" Max Sequence Length: {MAX_SEQ_LENGTH}")
print(f" Epochs: {EPOCHS}")
print(f" Output: {OUTPUT_DIR}")

# Step 1: Mount Google Drive
print("\n" + "="*70)
print(" Step 1: Checking Google Drive")
print("="*70)

if not Path('/content/drive').exists():
print("Mounting Google Drive...")
try:
from google.colab import drive
drive.mount('/content/drive')
print("[OK] Google Drive mounted")
except ImportError:
print("[WARNING] Not in Colab, skipping Drive mount")
else:
print("[OK] Google Drive already mounted")

# Step 2: Navigate and setup repository
print("\n" + "="*70)
print(" Step 2: Setting up Repository")
print("="*70)

project_dir = Path('/content/drive/MyDrive/slm_code')

if not project_dir.exists():
print("Cloning repository...")
os.chdir('/content/drive/MyDrive')
run_command(
'git clone https://github.com/sylvester-francis/slm-typescript-model.git slm_code',
'Cloning repository'
)
else:
print("Repository exists, updating...")
os.chdir(project_dir)
run_command('git pull origin main', 'Updating repository', check=False)

os.chdir(project_dir)
print(f"[OK] Working directory: {os.getcwd()}")

# Step 3: Install additional dependencies for 7B models
print("\n" + "="*70)
print(" Step 3: Installing Dependencies")
print("="*70)

# Core dependencies (required)
core_dependencies = [
"transformers>=4.36.0",
"peft>=0.7.0",
"trl>=0.7.0",
"accelerate>=0.25.0",
"bitsandbytes>=0.41.0",
]

# Optional: Flash Attention (faster but takes 10-20 min to compile)
# Set to False to skip and start training immediately
INSTALL_FLASH_ATTN = False # Change to True if you want faster training

for dep in core_dependencies:
run_command(f"pip install -q {dep}", f"Installing {dep.split('>=')[0]}", check=False)

if INSTALL_FLASH_ATTN:
print("\n Installing Flash Attention (this may take 10-20 minutes)...")
run_command("pip install -q flash-attn --no-build-isolation", "Installing Flash Attention", check=False)
else:
print("\n Skipping Flash Attention (training will be slightly slower but starts immediately)")

# Step 4: Check environment file
print("\n" + "="*70)
print(" Step 4: Checking Environment File")
print("="*70)

env_file = Path('.env')
if not env_file.exists():
print("[WARNING] .env file not found!")
try:
from google.colab import userdata
print("\nReading from Colab Secrets...")

github_token = userdata.get('GITHUB_TOKEN')
hf_token = userdata.get('HF_TOKEN')
so_key = userdata.get('STACKOVERFLOW_KEY') if 'STACKOVERFLOW_KEY' in userdata else ''

env_content = f"""# GitHub Configuration
GITHUB_TOKEN={github_token}

# StackOverflow Configuration (Optional)
STACKOVERFLOW_KEY={so_key}

# Hugging Face Configuration
HF_TOKEN={hf_token}
"""
with open('.env', 'w') as f:
f.write(env_content)
print("[OK] .env file created from Colab Secrets")

except Exception as e:
print(f"[ERROR] Could not read secrets: {e}")
print("Please add your tokens to Colab Secrets")
sys.exit(1)
else:
print("[OK] .env file found")

# Step 5: Check environment
print("\n" + "="*70)
print(" Step 5: Checking Environment")
print("="*70)

run_command('python -u scripts/check_environment.py', 'Environment check', check=False)

# Step 6: Create/update training script for 7B models
print("\n" + "="*70)
print(" Step 6: Preparing Training Configuration")
print("="*70)

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"[OK] Output directory ready: {OUTPUT_DIR}")

# Step 7: Train the model
print("\n" + "="*70)
print(" Step 7: Training 7B Model")
print("="*70)
print(f"\nEstimated time: ~2-3 hours on A100 for {DATASET}")
print(f"Training {model_config['description']}")

train_cmd = f"""python -u cli.py train \
--model {MODEL_NAME} \
--data {DATASET} \
--output {OUTPUT_DIR} \
--batch-size {BATCH_SIZE} \
--grad-accum {GRAD_ACCUM} \
--lora-r {LORA_R} \
--lora-alpha {LORA_ALPHA} \
--lr {LEARNING_RATE} \
--max-length {MAX_SEQ_LENGTH} \
--epochs {EPOCHS}"""

run_command(train_cmd, '7B Model training')

# Step 8: Evaluate the model
print("\n" + "="*70)
print(" Step 8: Evaluating Model")
print("="*70)

# Update evaluation to use the correct model path
eval_cmd = f"python -u cli.py evaluate --adapter {OUTPUT_DIR} --model {MODEL_NAME}"
run_command(eval_cmd, 'Model evaluation', check=False)

# Step 9: Create model card
print("\n" + "="*70)
print(" Step 9: Creating Model Card")
print("="*70)

model_card = f"""---
base_model: {MODEL_NAME}
library_name: peft
model_name: {HF_REPO_NAME}
tags:
- typescript
- code-generation
- react
- nextjs
- angular
- nodejs
- lora
- sft
- 7b
{'- reasoning' if MODEL_VARIANT == 'reasoning' else ''}
- transformers
- trl
license: mit
pipeline_tag: text-generation
language:
- en
---

# TypeScript SLM 7B{' - Reasoning Variant' if MODEL_VARIANT == 'reasoning' else ''}

{model_config['description']} for TypeScript code generation, optimized for React, Next.js, Angular, and Node.js.

## Model Details

- **Base Model**: [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})
- **Model Size**: 7B parameters
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Context Length**: {MAX_SEQ_LENGTH} tokens
- **LoRA Rank**: {LORA_R}
- **Training Dataset**: 5,000 high-quality TypeScript samples

{'## Reasoning Capabilities' if MODEL_VARIANT == 'reasoning' else ''}
{'This variant includes chain-of-thought reasoning for better code understanding and generation.' if MODEL_VARIANT == 'reasoning' else ''}

## Training Configuration

- Batch Size: {BATCH_SIZE}
- Gradient Accumulation: {GRAD_ACCUM}
- Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM}
- Learning Rate: {LEARNING_RATE}
- Epochs: {EPOCHS}
- Hardware: Google Colab A100 40GB

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "{MODEL_NAME}"
model = AutoModelForCausalLM.from_pretrained(
base_model,
device_map="auto",
torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "{HF_USERNAME}/{HF_REPO_NAME}")

# Generate code
prompt = "Write a React component with TypeScript:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

## Repository

https://github.com/sylvester-francis/slm-typescript-model
"""

model_card_path = Path(OUTPUT_DIR) / "README.md"
with open(model_card_path, 'w') as f:
f.write(model_card)
print(f"[OK] Model card created: {model_card_path}")

# Step 10: Upload to Hugging Face
print("\n" + "="*70)
print(" Step 10: Uploading to Hugging Face")
print("="*70)

upload_cmd = f"python -u cli.py upload --username {HF_USERNAME} --name {HF_REPO_NAME} --model {OUTPUT_DIR}"
run_command(upload_cmd, 'Hugging Face upload')

# Step 11: Backup to Google Drive
print("\n" + "="*70)
print(" Step 11: Backing up to Google Drive")
print("="*70)

backup_name = f"{HF_REPO_NAME}.tar.gz"
run_command(
f'tar -czf {backup_name} {OUTPUT_DIR}',
'Compressing model'
)
run_command(
f'cp {backup_name} /content/drive/MyDrive/',
'Copying to Google Drive'
)

# Final summary
print("\n" + "="*70)
print("[OK] 7B TRAINING PIPELINE COMPLETE!")
print("="*70)
print(f"\n Model Variant: {MODEL_VARIANT.upper()}")
print(f" Model uploaded to: https://huggingface.co/{HF_USERNAME}/{HF_REPO_NAME}")
print(f" Model backup: Google Drive/{backup_name}")
print(f" Model directory: {os.getcwd()}/{OUTPUT_DIR}")
print(f"\n Model ready for inference!")
print("="*70)

if __name__ == '__main__':
try:
main()
except KeyboardInterrupt:
print("\n\n[WARNING] Pipeline interrupted by user")
sys.exit(130)
except Exception as e:
print(f"\n\n[ERROR] Pipeline failed with error: {e}")
import traceback
traceback.print_exc()
sys.exit(1)
