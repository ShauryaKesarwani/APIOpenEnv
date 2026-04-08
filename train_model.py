"""
LoRA Fine-Tuning Script for Qwen 0.8B on API Workflow Tasks

This script trains a small model using trajectories from a larger expert model.
Uses Unsloth for efficient training with LoRA adapters.
"""

import os
import json
from typing import List, Dict
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-0.8B-Instruct"
OUTPUT_DIR = "./trained_model"
TRAJECTORIES_FILE = "data/qwen_trajectories.jsonl"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
LORA_ALPHA = 16
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
EPOCHS = 3


def load_trajectories(file_path: str) -> List[Dict]:
    """Load trajectories from JSONL file"""
    trajectories = []
    with open(file_path) as f:
        for line in f:
            trajectories.append(json.loads(line))
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


def prepare_dataset(trajectories: List[Dict]) -> Dataset:
    """Convert trajectories to HF dataset"""
    texts = []
    for traj in trajectories:
        # Only use successful trajectories
        if traj["final_result"]["grade"] >= 0.8:
            text = format_training_example(traj)
            texts.append({"text": text})
    
    print(f"Prepared {len(texts)} training examples from {len(trajectories)} trajectories")
    return Dataset.from_list(texts)


def main():
    """Main training function"""
    print("=" * 60)
    print("TRAINING QWEN 0.8B ON API WORKFLOW TASKS")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"LoRA rank: {LORA_RANK}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print()
    
    # Load trajectories
    print(f"Loading trajectories from {TRAJECTORIES_FILE}...")
    trajectories = load_trajectories(TRAJECTORIES_FILE)
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Prepare dataset
    print("\nPreparing training dataset...")
    dataset = prepare_dataset(trajectories)
    
    # Load model with Unsloth
    print(f"\nLoading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization
    )
    
    # Add LoRA adapters
    print("\nAdding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
    )
    
    # Training arguments
    print("\nSetting up training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=50,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
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
        max_seq_length=MAX_SEQ_LENGTH,
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
    
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("Training complete!")


if __name__ == "__main__":
    # Check if trajectories file exists
    if not os.path.exists(TRAJECTORIES_FILE):
        print(f"ERROR: Trajectories file not found: {TRAJECTORIES_FILE}")
        print("\nPlease run data collection first:")
        print("  python collect_trajectories.py --model qwen2.5:14b --episodes 100")
        exit(1)
    
    main()
