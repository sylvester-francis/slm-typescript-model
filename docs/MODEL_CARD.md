---
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
model_name: typescript-slm-1.5b
tags:
- typescript
- code-generation
- react
- nextjs
- angular
- nodejs
- lora
- sft
- transformers
- trl
license: mit
pipeline_tag: text-generation
language:
- en
datasets:
- custom
---

# TypeScript SLM 1.5B

A specialized Small Language Model for TypeScript code generation and understanding, optimized for React, Next.js, Angular, and Node.js frameworks.

## Model Description

This model is a fine-tuned version of [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) using LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning. It has been trained on 2,000-8,000 high-quality TypeScript code samples focusing on modern web development frameworks.

**Key Features:**
- Specialized in TypeScript and popular frameworks (React, Next.js, Angular, Node.js)
- Quality-scored training dataset with proper type annotations
- Optimized for code completion, generation, and understanding tasks
- Efficient inference with LoRA adapters

## Intended Use

### Primary Use Cases
- TypeScript code completion and generation
- React component scaffolding
- Next.js API route and page generation
- Angular service and directive creation
- Node.js/Express backend code generation
- Type definition and interface creation

### Out-of-Scope Use
- Production-critical code generation without human review
- Non-TypeScript/JavaScript code generation
- General-purpose text generation
- Code obfuscation or malicious code generation

## Training Data

The model was trained on a curated dataset of TypeScript code samples with the following distribution:

- **React** (43-58%): Components, hooks, context, custom hooks
- **Angular** (33-50%): Services, directives, modules, dependency injection
- **Next.js** (21-23%): Pages, API routes, SSR, SSG patterns
- **TypeScript** (9-16%): Advanced types, generics, utility types
- **Node.js** (6-11%): Express, NestJS, API servers

**Dataset Quality Scoring:**
- Samples scored 41-64 on quality metrics
- Prioritizes proper type annotations
- Excludes test files, debug code, and incomplete modules
- Focuses on production-quality patterns from popular repositories

## Training Procedure

### Training Hyperparameters

**Hardware:**
- Google Colab A100 40GB GPU
- CUDA acceleration with FP16 precision

**Configuration:**
- Base Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- Training Samples: 2,000-8,000 (depending on dataset tier)
- Epochs: 3
- Batch Size: 4
- Gradient Accumulation Steps: 8
- Effective Batch Size: 32
- Learning Rate: 2e-4
- Max Sequence Length: 1024
- LoRA Rank (r): 32
- LoRA Alpha: 16
- LoRA Dropout: 0.1
- Target Modules: All linear layers

**Training Time:**
- train_small.jsonl (2k samples): ~20-30 minutes on A100
- train_medium.jsonl (5k samples): ~50-75 minutes on A100
- train.jsonl (8k samples): ~2-3 hours on A100

## Usage

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
base_model,
device_map="auto",
torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "sylvester-francis/typescript-slm-1.5b")

# Generate code
prompt = """Write a React component that fetches user data and displays it in a card:

```typescript
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
**inputs,
max_new_tokens=256,
temperature=0.7,
do_sample=True,
top_p=0.95
)

generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_code)
```

### React Component Generation

```python
prompt = """Create a TypeScript React component with props for a user profile card:

```typescript
interface UserProfileProps {
"""

# Generate with the model...
```

### Next.js API Route

```python
prompt = """Write a Next.js API route for user authentication:

```typescript
// pages/api/auth/login.ts
"""

# Generate with the model...
```

### Angular Service

```python
prompt = """Create an Angular service for HTTP data fetching:

```typescript
import { Injectable } from '@angular/core';
"""

# Generate with the model...
```

## Performance

### Code Quality Metrics
- Proper TypeScript type annotations
- Framework-specific best practices
- Adherence to modern ES6+ patterns
- Clean, readable code structure

### Generation Speed
- Average: ~50-100 tokens/second on A100
- Latency: <100ms for typical completions
- Memory: ~3GB VRAM with adapter loaded

## Limitations

1. **Specialized Domain**: Works best for TypeScript and related frameworks. Performance degrades for other languages.

2. **Training Data Bias**: Reflects patterns from popular open-source repositories, which may not match all coding styles.

3. **Context Length**: Limited to 1024 tokens, which may be insufficient for very large files.

4. **No Real-time Updates**: Training data is static and doesn't include the latest framework versions or patterns.

5. **Requires Human Review**: Generated code should always be reviewed for security, correctness, and best practices.

6. **Type Safety**: While trained on typed code, generated types may not always be complete or optimal.

## Ethical Considerations

- **Code Licensing**: Ensure generated code complies with your project's license requirements
- **Security**: Always review generated code for security vulnerabilities
- **Testing**: Generated code should be thoroughly tested before production use
- **Attribution**: Consider the training data sources when using generated code commercially

## Training Infrastructure

**Software Stack:**
- PyTorch 2.9.0+cu126
- Transformers 4.57.2
- PEFT 0.18.0
- TRL 0.25.1
- Datasets 4.0.0
- bitsandbytes 0.41.0+

**Platform:**
- Google Colab Pro (recommended)
- Supports Mac M4 (MPS) for local training (slower)
- Compatible with T4, A100, and other CUDA GPUs

## Repository

Full training code, dataset filtering, and usage examples:
https://github.com/sylvester-francis/slm-typescript-model

## Model Card Authors

- Sylvester Francis (@sylvester-francis)

## Citations

### Base Model

```bibtex
@article{qwen2.5,
title={Qwen2.5-Coder Technical Report},
author={Qwen Team},
year={2024},
journal={arXiv preprint},
url={https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct}
}
```

### Training Framework

```bibtex
@misc{vonwerra2022trl,
title={{TRL: Transformer Reinforcement Learning}},
author={Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
year={2020},
journal={GitHub repository},
publisher={GitHub},
howpublished={\url{https://github.com/huggingface/trl}}
}
```

### LoRA

```bibtex
@article{hu2021lora,
title={LoRA: Low-Rank Adaptation of Large Language Models},
author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
journal={arXiv preprint arXiv:2106.09685},
year={2021}
}
```

## License

MIT License - See repository for full license text.

## Acknowledgments

- Built on Qwen 2.5 Coder by Alibaba Cloud
- Training powered by Hugging Face TRL and PEFT libraries
- Dataset curated from high-quality open-source TypeScript projects
