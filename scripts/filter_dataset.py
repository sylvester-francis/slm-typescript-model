#!/usr/bin/env python3
"""
Intelligent dataset filtering for TypeScript coding agent
Prioritizes high-quality, framework-specific TypeScript code
"""

import json
import re
from pathlib import Path

# Frameworks we care about for coding agent
PRIORITY_FRAMEWORKS = {
    'react': ['react', 'jsx', 'tsx', '@types/react'],
    'nextjs': ['next', 'nextjs', 'next.js', 'next/'],
    'vue': ['vue', '@vue/', 'nuxt'],
    'angular': ['angular', '@angular/'],
    'nestjs': ['nestjs', '@nestjs/'],
    'express': ['express', 'fastify', 'koa'],
    'typescript': ['typescript', 'ts-node', '@types/'],
}

# TypeScript-specific patterns (higher quality indicators)
TS_QUALITY_INDICATORS = [
    r'\binterface\s+\w+',  # Interface definitions
    r'\btype\s+\w+\s*=',   # Type aliases
    r'\benum\s+\w+',       # Enums
    r':\s*(string|number|boolean|void|any|unknown)',  # Type annotations
    r'<\w+>',              # Generics
    r'\bas\s+\w+',         # Type assertions
    r'\bimplements\s+',    # Class implements
    r'\bextends\s+',       # Class/interface extends
]

# Patterns to EXCLUDE (low quality)
EXCLUDE_PATTERNS = [
    r'^import.*from\s+["\']\.{1,2}/.*test',  # Test imports
    r'console\.log',                          # Debug code
    r'TODO|FIXME|XXX',                       # Unfinished code
    r'any\s*\[\]',                           # Untyped arrays
    r':\s*any\b',                            # Overuse of 'any'
]


def score_sample(data):
    """
    Score a sample based on quality and relevance
    Returns: (score, reasons)
    """
    text = data.get('text', '')
    text_lower = text.lower()
    path = data.get('path', '')
    repo = data.get('repo', '')

    score = 0
    reasons = []

    # 1. File type bonus
    if path.endswith('.tsx'):
        score += 10
        reasons.append('React TypeScript')
    elif path.endswith('.ts'):
        score += 5
        reasons.append('TypeScript')

    # 2. Framework detection (prioritize)
    for fw_name, keywords in PRIORITY_FRAMEWORKS.items():
        if any(kw in text_lower for kw in keywords):
            score += 15
            reasons.append(f'{fw_name.capitalize()} framework')
            break

    # 3. TypeScript quality indicators
    ts_features = 0
    for pattern in TS_QUALITY_INDICATORS:
        if re.search(pattern, text):
            ts_features += 1
    score += ts_features * 3
    if ts_features > 0:
        reasons.append(f'{ts_features} TS features')

    # 4. Exclude low-quality patterns
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            score -= 20
            reasons.append('Low quality pattern detected')

    # 5. Code complexity (good indicator)
    lines = text.split('\n')
    if 10 <= len(lines) <= 200:  # Sweet spot for examples
        score += 5
        reasons.append('Good length')
    elif len(lines) > 200:
        score -= 5  # Too long

    # 6. Has imports (real code, not snippets)
    if 'import ' in text and 'from ' in text:
        score += 5
        reasons.append('Complete module')

    # 7. Has exports (reusable code)
    if 'export ' in text:
        score += 3
        reasons.append('Exportable')

    # 8. Popular repos get bonus
    if repo:
        repo_lower = repo.lower()
        if any(x in repo_lower for x in ['react', 'next', 'vue', 'angular']):
            score += 10
            reasons.append('Popular repo')

    # 9. Penalize if too much 'any' type
    any_count = len(re.findall(r':\s*any\b', text))
    if any_count > 5:
        score -= any_count * 2
        reasons.append('Too many any types')

    return score, reasons


def filter_dataset(input_path, output_path, target_samples=8000, min_score=15):
    """
    Filter dataset to keep only high-quality TypeScript samples
    """
    print(f"Reading dataset from: {input_path}")

    # First pass: score all samples
    samples_with_scores = []
    total_read = 0

    with open(input_path, 'r') as f:
        for line in f:
            total_read += 1
            if total_read % 10000 == 0:
                print(f"  Scored {total_read} samples...")

            data = json.loads(line)
            score, reasons = score_sample(data)

            if score >= min_score:
                samples_with_scores.append((score, data, reasons))

    print(f"\nTotal samples read: {total_read}")
    print(f"Samples above threshold ({min_score}): {len(samples_with_scores)}")

    # Sort by score (best first)
    samples_with_scores.sort(reverse=True, key=lambda x: x[0])

    # Take top N samples
    selected = samples_with_scores[:target_samples]

    print(f"\nSelecting top {len(selected)} samples")
    print(f"Score range: {selected[-1][0]} to {selected[0][0]}")

    # Show some examples
    print("\nTop 5 samples:")
    for i, (score, data, reasons) in enumerate(selected[:5], 1):
        path = data.get('path', 'unknown')
        print(f"  {i}. Score: {score:3d} - {path}")
        print(f"     Reasons: {', '.join(reasons)}")

    # Write filtered dataset
    print(f"\nWriting to: {output_path}")
    with open(output_path, 'w') as f:
        for score, data, _ in selected:
            f.write(json.dumps(data) + '\n')

    print(f"✓ Filtered dataset created with {len(selected)} high-quality samples!")

    # Statistics
    frameworks = {}
    for _, data, _ in selected:
        text_lower = data['text'].lower()
        for fw_name, keywords in PRIORITY_FRAMEWORKS.items():
            if any(kw in text_lower for kw in keywords):
                frameworks[fw_name] = frameworks.get(fw_name, 0) + 1

    print("\nFramework distribution in filtered dataset:")
    for fw, count in sorted(frameworks.items(), key=lambda x: x[1], reverse=True):
        pct = count * 100 / len(selected)
        print(f"  {fw.capitalize():12s}: {count:5d} ({pct:5.1f}%)")


if __name__ == '__main__':
    import sys

    input_file = Path('data/processed/train.jsonl')
    output_file = Path('data/processed/train_ultra.jsonl')

    # Check if we want ultra-filtered version
    target = 3000  # Ultra-filtered: top 3k samples
    min_score = 35  # Much higher threshold

    if len(sys.argv) > 1:
        if sys.argv[1] == '--medium':
            target = 5000
            min_score = 25
            output_file = Path('data/processed/train_medium.jsonl')
        elif sys.argv[1] == '--small':
            target = 2000
            min_score = 40
            output_file = Path('data/processed/train_small.jsonl')

    filter_dataset(
        input_path=input_file,
        output_path=output_file,
        target_samples=target,
        min_score=min_score
    )
