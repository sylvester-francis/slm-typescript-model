import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "./typescript-slm-1.5b" # Path to saved adapter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(adapter_path=None):
    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    if adapter_path:
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
    return model, tokenizer

def generate_code(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_gen = len(outputs[0]) - len(inputs["input_ids"][0])
    speed = tokens_gen / (end_time - start_time)
    
    return generated_text, speed

def main():
    # Load model
    try:
        model, tokenizer = load_model(ADAPTER_PATH)
    except Exception as e:
        print(f"Could not load adapter (maybe not trained yet?). Loading base model only. Error: {e}")
        model, tokenizer = load_model()

    test_prompts = [
        "// Write a TypeScript function to calculate the Fibonacci sequence",
        "// Create a React component that displays a counter",
        "// Define a TypeScript interface for a User object with optional email",
        "// Write a function to fetch data from an API using async/await in TypeScript"
    ]
    
    print("\n--- Starting Evaluation ---")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        output, speed = generate_code(model, tokenizer, prompt)
        print("-" * 40)
        print(output)
        print("-" * 40)
        print(f"Speed: {speed:.2f} tokens/sec")

if __name__ == "__main__":
    main()
