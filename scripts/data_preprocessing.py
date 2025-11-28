import json
import os
from pathlib import Path
import re
from tqdm import tqdm
import random

# Configuration
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_code(content):
    """Basic code cleaning."""
    # Remove very long lines (likely minified)
    lines = content.split('\n')
    if any(len(line) > 1000 for line in lines):
        return None
        
    # Remove copyright headers (simple heuristic)
    # ... (implementation would go here, skipping for now to keep it simple)
    
    return content

def process_github_data():
    """Process GitHub raw data."""
    print("Processing GitHub data...")
    github_dir = DATA_DIR / "github"
    samples = []
    
    for file_path in github_dir.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                files = json.load(f)
                
            for file_data in files:
                content = file_data.get("content", "")
                cleaned = clean_code(content)
                if cleaned:
                    # Create an instruction-like format for code completion/infilling
                    # For base model training, we just want the raw code usually.
                    # But for instruction tuning, we might want to wrap it.
                    # For now, let's keep it as raw code for pre-training/fine-tuning
                    samples.append({
                        "text": cleaned,
                        "source": "github",
                        "repo": file_data.get("repo"),
                        "path": file_data.get("path")
                    })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return samples

def process_so_data():
    """Process StackOverflow data."""
    print("Processing StackOverflow data...")
    so_dir = DATA_DIR / "stackoverflow"
    samples = []
    
    for file_path in so_dir.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                qa_pairs = json.load(f)
                
            for qa in qa_pairs:
                # Format as an instruction
                # Remove HTML tags (simple regex)
                q_body = re.sub(r'<[^>]+>', '', qa["question_body"])
                a_body = re.sub(r'<[^>]+>', '', qa["answer_body"])
                
                text = f"User: {qa['title']}\n{q_body}\n\nAssistant: {a_body}"
                
                samples.append({
                    "text": text,
                    "source": "stackoverflow",
                    "tags": qa.get("tags")
                })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return samples

def main():
    all_samples = []
    all_samples.extend(process_github_data())
    all_samples.extend(process_so_data())
    
    print(f"Total samples before deduplication: {len(all_samples)}")
    
    # Simple deduplication by content hash
    seen = set()
    unique_samples = []
    for sample in all_samples:
        content_hash = hash(sample["text"])
        if content_hash not in seen:
            seen.add(content_hash)
            unique_samples.append(sample)
            
    print(f"Total samples after deduplication: {len(unique_samples)}")
    
    # Split train/val
    random.shuffle(unique_samples)
    split_idx = int(len(unique_samples) * 0.95)
    train_data = unique_samples[:split_idx]
    val_data = unique_samples[split_idx:]
    
    # Save as JSONL
    with open(OUTPUT_DIR / "train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open(OUTPUT_DIR / "validation.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(train_data)} training samples and {len(val_data)} validation samples.")

if __name__ == "__main__":
    main()
