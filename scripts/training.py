#!/usr/bin/env python3
"""
TypeScript SLM Training Script for Mac M4
Optimized for 24GB unified memory with MPS acceleration
"""

import os
import argparse
import logging
from datetime import datetime
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_device():
    """Detect and configure the appropriate device (MPS, CUDA, or CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("✓ Using Apple Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("⚠ No GPU detected, using CPU (will be slow!)")

    return device


def load_model_and_tokenizer(model_name, max_seq_length=1024):
    """Load model and tokenizer with appropriate settings"""
    logger.info(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_seq_length

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    logger.info(f"✓ Model loaded successfully")
    logger.info(f"✓ Tokenizer max length: {max_seq_length}")

    return model, tokenizer


def setup_lora(model, lora_r=64, lora_alpha=16, lora_dropout=0.1):
    """Configure LoRA for parameter-efficient fine-tuning"""
    logger.info("Setting up LoRA configuration")

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    logger.info(f"✓ LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    return peft_config


def load_training_data(data_path):
    """Load the training dataset"""
    logger.info(f"Loading dataset from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataset = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"✓ Loaded {len(dataset)} training samples")

    return dataset


def formatting_func(example):
    """Format dataset examples for training"""
    return example["text"]


def train(
    model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    data_path="data/processed/train.jsonl",
    output_dir="./models/typescript-slm-1.5b",
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_seq_length=1024,
    lora_r=64,
    save_steps=500,
    logging_steps=10,
    resume_from_checkpoint=None,
):
    """Main training function"""

    # Print configuration
    logger.info("="*70)
    logger.info("Training Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Dataset: {data_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Max sequence length: {max_seq_length}")
    logger.info(f"  LoRA rank: {lora_r}")
    logger.info("="*70)

    # Setup
    device = setup_device()
    model, tokenizer = load_model_and_tokenizer(model_name, max_seq_length)
    peft_config = setup_lora(model, lora_r=lora_r)
    dataset = load_training_data(data_path)

    # Training arguments optimized for Mac M4 24GB
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=True,  # Use BF16 for better precision on Apple Silicon
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        save_total_limit=3,  # Keep only last 3 checkpoints to save disk space
        logging_dir=f"{output_dir}/logs",
        report_to="none",  # Disable wandb by default
        gradient_checkpointing=True,
        dataloader_pin_memory=False,  # Set to False for MPS
    )

    # Initialize trainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
        formatting_func=formatting_func,
    )

    # Start training
    logger.info("\n" + "="*70)
    logger.info("Starting training...")
    logger.info("="*70 + "\n")

    start_time = datetime.now()

    try:
        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()

        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("\n" + "="*70)
        logger.info("✓ Training completed successfully!")
        logger.info(f"  Duration: {duration}")
        logger.info("="*70 + "\n")

    except KeyboardInterrupt:
        logger.warning("\n⚠ Training interrupted by user")
        logger.info("Saving current checkpoint...")
        trainer.save_model(f"{output_dir}/interrupted_checkpoint")
        logger.info(f"✓ Checkpoint saved to: {output_dir}/interrupted_checkpoint")
        return

    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        raise

    # Save final model
    logger.info("Saving final model...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✓ Model saved to: {output_dir}")

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("Training Summary:")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  Total epochs: {num_epochs}")
    logger.info(f"  Training time: {duration}")
    logger.info(f"  Model saved to: {output_dir}")
    logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Train TypeScript SLM on Mac M4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Base model name from Hugging Face"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/train.jsonl",
        help="Path to training data (JSONL file)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/typescript-slm-1.5b",
        help="Output directory for trained model"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device"
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank"
    )

    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Run training
    train(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
