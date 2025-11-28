# Model Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
NEW_MODEL_NAME = "typescript-slm-1.5b"

# LoRA Configuration
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Configuration
MAX_SEQ_LENGTH = 2048 # Reduced from 4096 for 4GB VRAM compatibility during inference, though training can be higher
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4 # Adjust based on VRAM/TPU memory
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 100
