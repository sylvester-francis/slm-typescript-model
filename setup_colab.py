#!/usr/bin/env python3
"""
Google Colab Setup Script for TypeScript SLM
Run this first in Colab to set up everything
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
"""Run a shell command and print status"""
print(f"\n{'='*70}")
print(f" {description}")
print(f"{'='*70}")
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
if result.returncode == 0:
print(f"[OK] Success")
if result.stdout:
print(result.stdout[:500]) # Print first 500 chars
else:
print(f"[ERROR] Failed: {result.stderr[:500]}")
return False
return True


def setup_colab():
"""Set up Colab environment for TypeScript SLM training"""

print("\n" + "="*70)
print(" TypeScript SLM - Google Colab Setup")
print("="*70)

# Check if we're in Colab
if 'COLAB_GPU' not in os.environ:
print("[WARNING] Warning: This doesn't appear to be Google Colab")
response = input("Continue anyway? (y/n): ")
if response.lower() != 'y':
return False

# Step 1: Mount Google Drive (if not already mounted)
print("\n Checking Google Drive...")
drive_path = Path('/content/drive')
if not drive_path.exists():
print("Mounting Google Drive...")
try:
from google.colab import drive
drive.mount('/content/drive')
print("[OK] Google Drive mounted")
except Exception as e:
print(f"[ERROR] Failed to mount drive: {e}")
return False
else:
print("[OK] Google Drive already mounted")

# Step 2: Check GPU
print("\n Checking GPU...")
try:
import torch
if torch.cuda.is_available():
gpu_name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"[OK] GPU: {gpu_name}")
print(f" VRAM: {vram:.1f} GB")
else:
print("[WARNING] No GPU detected!")
print(" Go to Runtime -> Change runtime type -> Hardware accelerator -> GPU")
return False
except ImportError:
print("[WARNING] PyTorch not installed")

# Step 3: Install/Upgrade dependencies
print("\n Installing dependencies...")
packages = [
'transformers>=4.35.0',
'datasets>=2.14.0',
'peft>=0.5.0',
'trl>=0.7.0',
'accelerate>=0.24.0',
'bitsandbytes>=0.41.0',
]

for package in packages:
cmd = f"pip install -q {package}"
run_command(cmd, f"Installing {package.split('>=')[0]}")

# Step 4: Clone/Update repository
print("\n Setting up repository...")
project_dir = Path('/content/drive/MyDrive/slm_code')

if project_dir.exists():
print(f"[OK] Project directory exists: {project_dir}")
os.chdir(project_dir)
run_command('git pull origin main', 'Updating from GitHub')
else:
print(f"Creating project directory: {project_dir}")
project_dir.parent.mkdir(parents=True, exist_ok=True)
os.chdir('/content/drive/MyDrive')
run_command(
'git clone https://github.com/sylvester-francis/slm-typescript-model.git slm_code',
'Cloning repository'
)
os.chdir(project_dir)

# Step 5: Create necessary directories
print("\n Creating directories...")
for dir_name in ['data/processed', 'models', 'logs']:
Path(dir_name).mkdir(parents=True, exist_ok=True)
print(f"[OK] {dir_name}/")

# Step 6: Generate filtered datasets if needed
print("\n Checking datasets...")
data_dir = Path('data/processed')
train_files = list(data_dir.glob('train*.jsonl'))

if not train_files:
print("[WARNING] No training data found")
print(" Please run data collection and preprocessing first")
else:
for f in sorted(train_files):
size_mb = f.stat().st_size / 1e6
print(f"[OK] {f.name}: {size_mb:.1f} MB")

# Step 7: Run environment check
print("\n Running environment check...")
if Path('scripts/check_environment.py').exists():
run_command('python -u scripts/check_environment.py', 'Environment check')

# Final summary
print("\n" + "="*70)
print("[OK] Setup complete! You can now run training:")
print("="*70)
print("\n[TIP] Quick start commands:")
print("\n# Fast training (2k samples, ~10-15 min):")
print("python cli.py train --data data/processed/train_small.jsonl --batch-size 32 --grad-accum 1")
print("\n# Medium training (5k samples, ~25-35 min):")
print("python cli.py train --data data/processed/train_medium.jsonl --batch-size 32 --grad-accum 1")
print("\n# Check training progress:")
print("tail -f training.log")
print("\n" + "="*70)

return True


if __name__ == '__main__':
success = setup_colab()
sys.exit(0 if success else 1)
