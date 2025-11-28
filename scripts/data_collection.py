import os
import requests
import git
import time
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
STACKOVERFLOW_KEY = os.getenv("STACKOVERFLOW_KEY")  # Optional, increases quota
DATA_DIR = Path("data/raw")
GITHUB_DIR = DATA_DIR / "github"
SO_DIR = DATA_DIR / "stackoverflow"

# Ensure directories exist
GITHUB_DIR.mkdir(parents=True, exist_ok=True)
SO_DIR.mkdir(parents=True, exist_ok=True)

# Frameworks to target
FRAMEWORKS = [
    "react", "next.js", "angular", "vue", "svelte", 
    "express", "nestjs", "fastify", "prisma", "typeorm"
]

def search_github_repos(query, min_stars=1000, limit=20):
    """Search for TypeScript repositories on GitHub."""
    print(f"Searching GitHub for {query}...")
    url = "https://api.github.com/search/repositories"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    params = {
        "q": f"{query} language:TypeScript stars:>{min_stars}",
        "sort": "stars",
        "order": "desc",
        "per_page": limit
    }
    
    repos = []
    page = 1
    while len(repos) < limit:
        params["page"] = page
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Error searching GitHub: {response.status_code} - {response.text}")
            break
            
        data = response.json()
        items = data.get("items", [])
        if not items:
            break
            
        repos.extend(items)
        page += 1
        time.sleep(1) # Rate limiting
        
    return repos[:limit]

def clone_and_extract_ts(repo_url, repo_name):
    """Clone a repo and extract TypeScript files."""
    repo_path = GITHUB_DIR / "temp" / repo_name
    if repo_path.exists():
        shutil.rmtree(repo_path)
    
    print(f"Cloning {repo_name}...")
    try:
        git.Repo.clone_from(repo_url, repo_path, depth=1)
        
        # Extract .ts and .tsx files
        ts_files = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith((".ts", ".tsx")) and not file.endswith(".d.ts"):
                    full_path = Path(root) / file
                    try:
                        content = full_path.read_text(encoding="utf-8")
                        if len(content.strip()) > 50: # Skip empty/tiny files
                            ts_files.append({
                                "repo": repo_name,
                                "path": str(full_path.relative_to(repo_path)),
                                "content": content
                            })
                    except Exception as e:
                        continue # Skip binary or unreadable files
        
        # Save extracted files
        output_file = GITHUB_DIR / f"{repo_name.replace('/', '_')}.json"
        with open(output_file, "w") as f:
            json.dump(ts_files, f, indent=2)
            
        print(f"Saved {len(ts_files)} TypeScript files from {repo_name}")
        
    except Exception as e:
        print(f"Failed to clone/process {repo_name}: {e}")
    finally:
        if repo_path.exists():
            shutil.rmtree(repo_path)

def fetch_stackoverflow_questions(tag, limit=100):
    """Fetch TypeScript questions and answers from StackOverflow."""
    print(f"Fetching StackOverflow questions for tag: {tag}...")
    url = "https://api.stackexchange.com/2.3/questions"
    params = {
        "order": "desc",
        "sort": "votes",
        "tagged": tag,
        "site": "stackoverflow",
        "pagesize": 100,
        "filter": "withbody", # Include body
        "key": STACKOVERFLOW_KEY
    }
    
    questions = []
    page = 1
    
    while len(questions) < limit:
        params["page"] = page
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching SO: {response.status_code}")
            break
            
        data = response.json()
        items = data.get("items", [])
        if not items:
            break
            
        # For each question, we need the accepted answer if it exists
        for item in items:
            if "accepted_answer_id" in item:
                questions.append(item)
                
        page += 1
        time.sleep(0.5) # Rate limiting
        
    # Now fetch answers
    results = []
    for q in tqdm(questions[:limit], desc="Fetching answers"):
        ans_id = q["accepted_answer_id"]
        ans_url = f"https://api.stackexchange.com/2.3/answers/{ans_id}"
        ans_params = {
            "site": "stackoverflow",
            "filter": "withbody",
            "key": STACKOVERFLOW_KEY
        }
        ans_resp = requests.get(ans_url, params=ans_params)
        if ans_resp.status_code == 200:
            ans_data = ans_resp.json().get("items", [])
            if ans_data:
                answer = ans_data[0]
                results.append({
                    "title": q["title"],
                    "question_body": q["body"],
                    "answer_body": answer["body"],
                    "tags": q["tags"],
                    "score": q["score"],
                    "link": q["link"]
                })
        time.sleep(0.2)
        
    output_file = SO_DIR / f"so_{tag}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} Q&A pairs for {tag}")

def main():
    # 1. Collect from GitHub
    print("--- Starting GitHub Collection ---")
    # Search for generic typescript repos first
    repos = search_github_repos("language:TypeScript", limit=5)
    
    # Add framework specific searches
    for framework in FRAMEWORKS:
        repos.extend(search_github_repos(f"{framework} language:TypeScript", limit=3))
    
    # Deduplicate repos
    unique_repos = {r["full_name"]: r for r in repos}.values()
    
    print(f"Found {len(unique_repos)} unique repositories to process.")
    
    for repo in tqdm(unique_repos, desc="Processing Repos"):
        clone_and_extract_ts(repo["clone_url"], repo["full_name"])

    # 2. Collect from StackOverflow
    print("\n--- Starting StackOverflow Collection ---")
    fetch_stackoverflow_questions("typescript", limit=50)
    for framework in FRAMEWORKS[:3]: # Limit to top 3 to save time/quota
        fetch_stackoverflow_questions(f"typescript+{framework}", limit=30)

if __name__ == "__main__":
    main()
