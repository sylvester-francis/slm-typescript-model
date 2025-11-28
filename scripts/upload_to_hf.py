import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv

load_dotenv()

def upload_model(model_path: str = "./models/typescript-slm-1.5b",
username: str = None,
model_name: str = "typescript-slm-1.5b"):
"""
Upload trained model to Hugging Face Hub.

Args:
model_path: Path to the model directory
username: Hugging Face username
model_name: Name for the model on Hugging Face
"""
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
raise ValueError("Error: HF_TOKEN not found in environment variables. Please set it in .env file.")

if not username:
raise ValueError("Error: username is required. Please provide your Hugging Face username.")

model_path = Path(model_path)
if not model_path.exists():
raise ValueError(f"Error: Model directory not found at {model_path}")

repo_id = f"{username}/{model_name}"
print(f"Uploading model from {model_path} to {repo_id}...")

try:
# Create repository
print(f"Creating repository {repo_id}...")
create_repo(repo_id, token=HF_TOKEN, exist_ok=True, repo_type="model")
print("[OK] Repository created/verified")

# Upload model files
print(f"Uploading files from {model_path}...")
upload_folder(
folder_path=str(model_path),
repo_id=repo_id,
repo_type="model",
token=HF_TOKEN,
ignore_patterns=["*.pyc", "__pycache__", ".git", ".gitignore"]
)
print(f"[OK] Upload complete!")
print(f"\nModel available at: https://huggingface.co/{repo_id}")

except Exception as e:
raise Exception(f"Upload failed: {e}")

def main():
"""Legacy main function for backward compatibility"""
upload_model()

if __name__ == "__main__":
import sys
if len(sys.argv) > 1:
username = sys.argv[1]
model_name = sys.argv[2] if len(sys.argv) > 2 else "typescript-slm-1.5b"
upload_model(username=username, model_name=model_name)
else:
print("Usage: python upload_to_hf.py <username> [model_name]")
sys.exit(1)
