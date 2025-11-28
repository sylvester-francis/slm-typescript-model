#!/usr/bin/env python3
"""
Complete Colab Training and Upload Script
This script automates the entire training pipeline on Google Colab
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a shell command and print status"""
    print(f"\n{'='*70}")
    print(f"🔧 {description}")
    print(f"{'='*70}")
    print(f"Running: {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    if check and result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        sys.exit(1)

    print(f"\n✅ Success: {description}")
    return result.returncode == 0

def main():
    print("\n" + "="*70)
    print("🚀 TypeScript SLM - Complete Colab Training Pipeline")
    print("="*70)

    # Configuration
    HF_USERNAME = "sylvester-francis"
    DATASET = "data/processed/train_small.jsonl"  # Change to train_medium.jsonl or train.jsonl for larger datasets
    BATCH_SIZE = 4
    GRAD_ACCUM = 8
    LORA_R = 32
    EPOCHS = 3

    print(f"\nConfiguration:")
    print(f"  HF Username: {HF_USERNAME}")
    print(f"  Dataset: {DATASET}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Gradient Accumulation: {GRAD_ACCUM}")
    print(f"  LoRA Rank: {LORA_R}")
    print(f"  Epochs: {EPOCHS}")

    # Step 1: Mount Google Drive (if in Colab)
    print("\n" + "="*70)
    print("📁 Step 1: Checking Google Drive")
    print("="*70)

    if not Path('/content/drive').exists():
        print("Mounting Google Drive...")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✅ Google Drive mounted")
        except ImportError:
            print("⚠️  Not in Colab, skipping Drive mount")
    else:
        print("✅ Google Drive already mounted")

    # Step 2: Navigate and setup repository
    print("\n" + "="*70)
    print("📥 Step 2: Setting up Repository")
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
    print(f"✅ Working directory: {os.getcwd()}")

    # Step 3: Run setup
    print("\n" + "="*70)
    print("🔧 Step 3: Running Setup")
    print("="*70)

    run_command('python setup_colab.py', 'Colab environment setup')

    # Step 4: Create .env file
    print("\n" + "="*70)
    print("🔑 Step 4: Checking Environment File")
    print("="*70)

    # Check if .env exists
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  .env file not found!")
        print("Please create a .env file with your tokens:")
        print("""
# GitHub Configuration
GITHUB_TOKEN=your_github_token

# StackOverflow Configuration (Optional)
STACKOVERFLOW_KEY=your_stackoverflow_key

# Hugging Face Configuration
HF_TOKEN=your_huggingface_token
        """)

        # Prompt user for tokens in Colab
        try:
            from google.colab import userdata
            print("\nAttempting to read from Colab Secrets...")

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
            print("✅ .env file created from Colab Secrets")

        except ImportError:
            print("❌ Not in Colab. Please create .env file manually and rerun.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Could not read secrets: {e}")
            print("Please add your tokens to Colab Secrets:")
            print("  1. Click the key icon 🔑 in the left sidebar")
            print("  2. Add secrets: GITHUB_TOKEN, HF_TOKEN, STACKOVERFLOW_KEY")
            sys.exit(1)
    else:
        print("✅ .env file found")

    # Step 5: Check environment
    print("\n" + "="*70)
    print("🔬 Step 5: Checking Environment")
    print("="*70)

    run_command('python scripts/check_environment.py', 'Environment check', check=False)

    # Step 6: Train the model
    print("\n" + "="*70)
    print("🎯 Step 6: Training Model")
    print("="*70)
    print(f"\nThis will take approximately 20-30 minutes on A100...")
    print(f"Training with {DATASET}")

    train_cmd = f"""python cli.py train \
  --data {DATASET} \
  --batch-size {BATCH_SIZE} \
  --grad-accum {GRAD_ACCUM} \
  --lora-r {LORA_R} \
  --epochs {EPOCHS}"""

    run_command(train_cmd, 'Model training')

    # Step 7: Evaluate the model
    print("\n" + "="*70)
    print("📊 Step 7: Evaluating Model")
    print("="*70)

    run_command('python cli.py evaluate', 'Model evaluation', check=False)

    # Step 8: Upload to Hugging Face
    print("\n" + "="*70)
    print("☁️  Step 8: Uploading to Hugging Face")
    print("="*70)

    upload_cmd = f"python cli.py upload --username {HF_USERNAME}"
    run_command(upload_cmd, 'Hugging Face upload')

    # Step 9: Backup to Google Drive
    print("\n" + "="*70)
    print("💾 Step 9: Backing up to Google Drive")
    print("="*70)

    run_command(
        'tar -czf typescript-slm-model.tar.gz models/typescript-slm-1.5b/',
        'Compressing model'
    )
    run_command(
        'cp typescript-slm-model.tar.gz /content/drive/MyDrive/',
        'Copying to Google Drive'
    )

    # Final summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n📦 Model uploaded to: https://huggingface.co/{HF_USERNAME}/typescript-slm-1.5b")
    print(f"💾 Model backup saved to: Google Drive/typescript-slm-model.tar.gz")
    print(f"📁 Model directory: {os.getcwd()}/models/typescript-slm-1.5b/")
    print("\n" + "="*70)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
