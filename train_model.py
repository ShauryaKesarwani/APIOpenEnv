"""
LoRA Fine-Tuning Script for Qwen 0.8B on API Workflow Tasks

This script trains a small model using trajectories from a larger expert model.
Uses Unsloth for efficient training with LoRA adapters.
"""

import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import torch

load_dotenv(dotenv_path=".env")

# Configuration (override via CLI args)
BASE_MODEL = "Qwen/Qwen2.5-0.8B-Instruct"
OUTPUT_DIR = "./trained_model"
TRAJECTORIES_FILE = "data\\qwen_trajectories.jsonl"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
LORA_ALPHA = 16
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
EPOCHS = 3
MIN_GRADE = 0.8
MAX_TRAJECTORIES = None


def load_trajectories(file_path: str, max_trajectories: int | None = None) -> List[Dict]:
    """Load trajectories from JSONL file."""
    trajectories: List[Dict] = []
    with open(file_path) as f:
        for line in f:
            trajectories.append(json.loads(line))
            if max_trajectories and len(trajectories) >= max_trajectories:
                break
    return trajectories


def format_training_example(trajectory: Dict) -> str:
    """Format a trajectory into a training text"""
    # Build conversation format
    text_parts = []
    
    # System prompt
    text_parts.append("<|im_start|>system\nYou are an AI agent that calls APIs to complete tasks.<|im_end|>")
    
    # Each step is a user message (state) + assistant message (action)
    for step in trajectory["steps"]:
        obs = step["observation"]
        action = step["action"]
        
        # User: current state
        user_msg = f"Task: {obs['task']}\nAvailable APIs: {', '.join(obs['available_apis'])}"
        if obs.get('api_call_history'):
            user_msg += f"\nPrevious calls: {len(obs['api_call_history'])}"
        
        text_parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
        
        # Assistant: action
        action_json = json.dumps({"api_name": action["api_name"], "args": action["args"]})
        text_parts.append(f"<|im_start|>assistant\n{action_json}<|im_end|>")
    
    return "\n".join(text_parts)


def prepare_dataset(trajectories: List[Dict], min_grade: float = 0.8) -> Dataset:
    """Convert trajectories to HF dataset."""
    texts = []
    for traj in trajectories:
        if traj.get("final_result", {}).get("grade", 0.0) >= min_grade:
            text = format_training_example(traj)
            texts.append({"text": text})

    print(f"Prepared {len(texts)} training examples from {len(trajectories)} trajectories")
    return Dataset.from_list(texts)


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="LoRA fine-tune a small model on collected trajectories")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--trajectories", type=str, default=TRAJECTORIES_FILE)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--min-grade", type=float, default=MIN_GRADE)
    parser.add_argument("--max-trajectories", type=int, default=MAX_TRAJECTORIES)

    # Performance/footprint knobs (useful for short runs)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast 30-min-ish sanity run (reduces seq len, LoRA rank, etc.)",
    )

    args = parser.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 1)
        args.max_trajectories = args.max_trajectories or 30
        args.min_grade = min(args.min_grade, 0.6)
        args.max_seq_length = min(args.max_seq_length, 1024)
        args.batch_size = min(args.batch_size, 1)
        args.grad_accum = max(args.grad_accum, 8)
        args.warmup_steps = min(args.warmup_steps, 10)
        args.lora_rank = min(args.lora_rank, 8)
        args.lora_alpha = min(args.lora_alpha, 8)
        args.learning_rate = max(args.learning_rate, 1e-4)

    print("=" * 60)
    print("TRAINING (LoRA) ON API WORKFLOW TASKS")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"Batch size: {args.batch_size} (grad accum: {args.grad_accum})")
    print(f"Epochs: {args.epochs}")
    print(f"Min grade: {args.min_grade}")
    if args.max_trajectories:
        print(f"Max trajectories: {args.max_trajectories}")
    print()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. LoRA training via Unsloth typically requires a GPU. "
            "If you only want to collect data, skip train_model.py."
        )

    # Load trajectories
    print(f"Loading trajectories from {args.trajectories}...")
    trajectories = load_trajectories(args.trajectories, max_trajectories=args.max_trajectories)
    print(f"Loaded {len(trajectories)} trajectories")

    # Prepare dataset
    print("\nPreparing training dataset...")
    dataset = prepare_dataset(trajectories, min_grade=args.min_grade)
    if len(dataset) == 0:
        raise ValueError(
            "No training examples after filtering. "
            "Lower --min-grade or collect more trajectories."
        )

    # Load model with Unsloth
    print(f"\nLoading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization
    )
    
    # Add LoRA adapters
    print("\nAdding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
    )
    
    # Training arguments
    print("\nSetting up training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        save_total_limit=2,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )
    
    # Train!
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print()
    
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nModel saved to: {args.output_dir}")
    print("Training complete!")


if __name__ == "__main__":
    # Note: CLI args are parsed inside main().
    if not os.path.exists(TRAJECTORIES_FILE):
        print(f"WARNING: Default trajectories file not found: {TRAJECTORIES_FILE}")
        print("You can pass a different file with: --trajectories <path>")

    main()
