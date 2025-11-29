"""
Google Colab Quick Start Cell
Copy and paste this entire cell into Colab to start training immediately
"""

# ============================================================================
# TypeScript SLM - Google Colab Quick Start
# ============================================================================

print(" TypeScript SLM - Quick Start Setup")
print("="*70)

# Step 1: Mount Google Drive
print("\n Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
print("[OK] Drive mounted")

# Step 2: Navigate to project directory
import os
from pathlib import Path

project_dir = Path('/content/drive/MyDrive/slm_code')

if not project_dir.exists():
print("\n Cloning repository (first time)...")
os.chdir('/content/drive/MyDrive')
!git clone https://github.com/sylvester-francis/slm-typescript-model.git slm_code
print("[OK] Repository cloned")
else:
print("\n Updating repository...")
os.chdir(project_dir)
!git pull origin main
print("[OK] Repository updated")

os.chdir(project_dir)
print(f"[OK] Working directory: {os.getcwd()}")

# Step 3: Check GPU
print("\n Checking GPU...")
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
print("[OK] GPU check complete")

# Step 4: Verify tokens are in Colab Secrets
print("\n Checking Colab Secrets...")
try:
from google.colab import userdata
required_secrets = ['GITHUB_TOKEN', 'HF_TOKEN']
missing = []

for secret in required_secrets:
try:
userdata.get(secret)
print(f" [OK] {secret} found")
except:
missing.append(secret)
print(f" [ERROR] {secret} missing")

if missing:
print(f"\n[WARNING] Missing secrets: {', '.join(missing)}")
print(" Add them by clicking the icon in the left sidebar")
else:
print("[OK] All secrets configured")
except ImportError:
print("[WARNING] Not in Colab - secrets check skipped")

# Step 5: Display training options
print("\n" + "="*70)
print(" Ready to Train! Choose your model:")
print("="*70)
print("\n1. 1.5B Model (Fastest - 20-30 min on A100)")
print(" !python colab_train_and_upload.py")
print("\n2. 7B Standard Model (Best Quality - 2-3 hours on A100)")
print(" !python colab_train_7b.py")
print("\n3. 7B Reasoning Model (Advanced - 2-3 hours on A100)")
print(" First edit: !sed -i 's/MODEL_VARIANT = \"standard\"/MODEL_VARIANT = \"reasoning\"/' colab_train_7b.py")
print(" Then run: !python colab_train_7b.py")
print("\n" + "="*70)
print("[TIP] Tip: Run one of the commands above in the next cell")
print("="*70)
