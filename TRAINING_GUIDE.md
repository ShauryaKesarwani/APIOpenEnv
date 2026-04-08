# Training Guide for 0.8B API Prediction Model

## Overview
This guide explains how to train a small 0.8B parameter model to predict correct API usage sequences using RLHF and fine-tuning.

## Architecture Changes Made

### 1. **Tool Calling Format** ✅
- Converted from JSON prompting to OpenAI-compatible tool calling format
- Each API is now defined as a structured function with parameters
- Better for training smaller models on function calling
- More standardized format for LLM evaluation

### 2. **1-Second Delay** ✅
- Added `time.sleep(1.0)` after each task iteration
- Prevents server overload during evaluation

### 3. **Backward Compatibility**
- Use `--no-tools` flag to use legacy JSON format
- Default is now tool calling format

## Training Pipeline

### Phase 1: Data Collection with Qwen (Upper Bound)

**Goal:** Collect high-quality trajectories using Qwen as your expert model

```bash
# Test with Qwen first to establish upper bound
export OPENAI_API_KEY="your-ollama-key"
export INFERENCE_SERVER="http://localhost:11434/v1"  # Ollama server
export MODEL_LOWER_NAME="qwen2.5:14b"  # or your Qwen variant

# Run evaluation to collect trajectories
python baseline_agent.py \
  --model qwen2.5:14b \
  --episodes 20 \
  --difficulty all \
  --output data/qwen_trajectories.json
```

**What to collect:**
- Successful task completions (grade > 0.8)
- API call sequences
- Tool call format with parameters
- Reasoning traces (if using verbose mode)

### Phase 2: Dataset Preparation

**Format for Fine-Tuning:**

Create training dataset in conversational format with tool calls:

```jsonl
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI agent that completes tasks by calling APIs..."
    },
    {
      "role": "user",
      "content": "TASK: Create invoice for user U101's order O501\nAVAILABLE APIs: get_user, get_orders, create_invoice\nSTEP: 1/5\n..."
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "get_user",
            "arguments": "{\"user_id\": \"U101\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "content": "{\"success\": true, \"data\": {...}}"
    },
    ...
  ]
}
```

**Create Dataset Builder:**

```python
# data_builder.py
import json

def build_training_data(trajectories_file, output_file):
    """Convert Qwen trajectories to fine-tuning format"""
    with open(trajectories_file) as f:
        data = json.load(f)
    
    training_examples = []
    for episode in data['episodes']:
        if episode['grade'] < 0.8:  # Only use successful episodes
            continue
        
        # Extract API call history and convert to conversational format
        # ... (implementation details)
    
    with open(output_file, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
```

### Phase 3: Supervised Fine-Tuning (SFT)

**Option A: Using Ollama + Unsloth (Recommended for 0.8B)**

```bash
# Install unsloth for efficient training
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# Training script
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Load 0.8B base model (e.g., Qwen2.5-0.8B, Phi-2, etc.)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-0.8B",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Add LoRA adapters for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
)

# Training
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 50,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        output_dir = "outputs/sft_model",
    ),
)

trainer.train()
model.save_pretrained("models/api_agent_0.8b_sft")
```

**Option B: Using OpenAI Fine-Tuning API**

```bash
# Prepare data in OpenAI format
openai tools fine_tunes.prepare_data -f data/training_data.jsonl

# Upload and start fine-tuning
openai api fine_tunes.create \
  -t data/training_data_prepared.jsonl \
  -m gpt-3.5-turbo \
  --n_epochs 3 \
  --batch_size 4 \
  --learning_rate_multiplier 0.1
```

### Phase 4: Reinforcement Learning with Human Feedback (RLHF)

**Option A: Direct Preference Optimization (DPO) - Simpler**

DPO is easier to implement than PPO and works well for smaller models:

```python
from trl import DPOTrainer

# Create preference pairs from your data
# Preferred: successful trajectories (grade > 0.8)
# Rejected: failed trajectories or inefficient sequences

preference_data = [
    {
        "prompt": "TASK: Create invoice for user U101...",
        "chosen": "get_user(U101)",  # Correct first step
        "rejected": "create_invoice(U101, O501)"  # Wrong (missing prerequisite)
    },
    ...
]

# DPO Training
dpo_trainer = DPOTrainer(
    model=sft_model,
    ref_model=sft_model,  # Reference model
    beta=0.1,  # KL penalty coefficient
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=5e-5,
        output_dir="outputs/dpo_model",
    ),
)

dpo_trainer.train()
```

**Option B: PPO with Reward Model (Advanced)**

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Define reward function based on environment feedback
def reward_function(trajectory, env_result):
    """
    Reward based on:
    - Task completion: +10
    - Correct API sequence: +2 per correct call
    - Minimal steps: +5 if completed under max_steps/2
    - Failed calls: -1 per failure
    """
    reward = 0
    if env_result['completed']:
        reward += 10
    reward += env_result['grade'] * 5
    reward -= len([c for c in trajectory if not c['success']])
    return reward

# PPO Config
ppo_config = PPOConfig(
    model_name="models/api_agent_0.8b_sft",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
)

# Train with PPO
model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
ppo_trainer = PPOTrainer(ppo_config, model, tokenizer=tokenizer)

# Training loop
for epoch in range(num_epochs):
    for batch in dataset:
        # Generate responses
        response = ppo_trainer.generate(batch["query"])
        
        # Get rewards from environment
        rewards = [reward_function(r, env.step(r)) for r in response]
        
        # PPO update
        stats = ppo_trainer.step(batch["query"], response, rewards)
```

### Phase 5: Automatic Learning with Reward System

**Self-Play Training Loop:**

```python
# auto_train.py
import time
from baseline_agent import BaselineAgent
from server.api_open_env_environment import ApiOpenEnvironment

def self_play_training(
    model_path: str,
    num_iterations: int = 1000,
    save_interval: int = 100
):
    """
    Continuously improve model through self-play with environment feedback
    """
    agent = BaselineAgent(model=model_path, use_tools=True)
    env = ApiOpenEnvironment()
    
    successful_trajectories = []
    failed_trajectories = []
    
    for iteration in range(num_iterations):
        # Run episode
        obs = env.reset()
        trajectory = []
        
        while not obs.done:
            action = agent.get_action(obs)
            obs = env.step(action)
            
            trajectory.append({
                'state': obs,
                'action': action,
                'reward': obs.reward,
                'success': obs.last_api_result.get('success', False)
            })
            
            time.sleep(1.0)  # Prevent overload
        
        # Categorize trajectory
        grade = env.grade()
        if grade > 0.8:
            successful_trajectories.append(trajectory)
        else:
            failed_trajectories.append(trajectory)
        
        # Periodic retraining
        if iteration % save_interval == 0:
            print(f"Iteration {iteration}: Retraining with {len(successful_trajectories)} good examples")
            
            # Create preference pairs for DPO
            preference_data = create_preference_pairs(
                successful_trajectories,
                failed_trajectories
            )
            
            # Retrain model
            model = retrain_with_dpo(model_path, preference_data)
            agent = BaselineAgent(model=model_path, use_tools=True)
            
            # Clear old trajectories to focus on recent performance
            successful_trajectories = successful_trajectories[-500:]
            failed_trajectories = failed_trajectories[-500:]

if __name__ == "__main__":
    self_play_training("models/api_agent_0.8b_sft", num_iterations=5000)
```

## Evaluation & Comparison

### Compare Models:

```bash
# 1. Qwen (Upper Bound)
python baseline_agent.py --model qwen2.5:14b --episodes 10 --output results/qwen.json

# 2. Your 0.8B SFT model
python baseline_agent.py --model models/api_agent_0.8b_sft --episodes 10 --output results/sft.json

# 3. Your 0.8B DPO model
python baseline_agent.py --model models/api_agent_0.8b_dpo --episodes 10 --output results/dpo.json

# Compare results
python compare_results.py results/qwen.json results/sft.json results/dpo.json
```

### Key Metrics to Track:

1. **Completion Rate**: % of tasks completed successfully
2. **Average Grade**: Quality of solutions (0-1)
3. **Efficiency**: Average steps to completion
4. **API Accuracy**: % of correct API calls
5. **Parameter Efficiency**: Performance per billion parameters

**Target Goals for 0.8B Model:**
- Completion Rate: 70%+ of Qwen's rate
- Average Grade: 0.6+ (compared to Qwen's ~0.85)
- Efficiency: Within 20% of Qwen's step count

## Best Practices

1. **Start Simple**: Begin with SFT on successful trajectories
2. **Iterate**: Use DPO for preference learning before attempting PPO
3. **Monitor Overfitting**: Keep validation set, track generalization
4. **Curriculum Learning**: Start with easy tasks, gradually add medium/hard
5. **Regular Evaluation**: Test on held-out tasks frequently
6. **Ablation Studies**: Test impact of each training phase

## Tools & Libraries

- **Unsloth**: Fast, memory-efficient training for small models
- **TRL (Transformer Reinforcement Learning)**: RLHF/DPO/PPO implementations
- **vLLM**: Fast inference for evaluation
- **Ollama**: Local model serving
- **Weights & Biases**: Experiment tracking

## Expected Timeline

1. Data Collection (Qwen): 1-2 days
2. SFT Training: 2-4 hours (with LoRA on single GPU)
3. DPO Training: 1-2 hours
4. Evaluation & Iteration: Ongoing
5. Self-Play Training: Continuous improvement

## Cost Considerations

- Using Ollama locally: **Free** (requires GPU with 8GB+ VRAM for 0.8B)
- Cloud training (Colab Pro): ~$10-20/month
- Commercial APIs (OpenAI fine-tuning): ~$50-100 for full pipeline

## Next Steps

1. ✅ Test current implementation with Qwen
2. ✅ Collect 200+ successful trajectories
3. Convert to training format
4. Run SFT on 0.8B base model
5. Evaluate and iterate
6. Apply DPO/PPO for further improvement
7. Deploy best model and compare to Qwen baseline
