import os
from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "typescript-slm-1.5b"
USERNAME = "YOUR_USERNAME" # Change this

def main():
    if not HF_TOKEN:
        print("Error: HF_TOKEN not found in environment variables.")
        return

    repo_id = f"{USERNAME}/{MODEL_NAME}"
    print(f"Uploading to {repo_id}...")
    
    try:
        create_repo(repo_id, token=HF_TOKEN, exist_ok=True)
        
        upload_folder(
            folder_path=MODEL_NAME,
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN
        )
        print("Upload complete!")
        
    except Exception as e:
        print(f"Error uploading: {e}")

if __name__ == "__main__":
    main()
