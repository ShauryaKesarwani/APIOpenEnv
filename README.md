---
title: API Workflow Environment
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - hackathon
  - reinforcement-learning
---

# API Workflow Environment 🚀

**Scaler x Meta x PyTorch OpenEnv Hackathon Project**

A real-world OpenEnv environment where AI agents learn to orchestrate multi-step API workflows. Unlike toy problems, this simulates actual backend operations like invoice generation and support ticket resolution.

## The Challenge

The agent must:
- Decide which API to call
- Pass correct arguments based on previous results
- Complete workflows in minimal steps
- Handle edge cases (e.g., refunds denied after 30 days)

## Three Difficulty Levels

| Level | Task | APIs | Max Steps |
|-------|------|------|-----------|
| **Easy** | Fetch user email | `get_user` | 3 |
| **Medium** | Generate invoice | `get_user` → `get_orders` → `get_product` → `create_invoice` | 8 |
| **Hard** | Resolve support ticket | `get_ticket` → `get_user` → `get_orders` → `process_refund` → `send_email` | 10 |

## Quick Start

### 1. Run the Tests

```bash
uv run python test_env.py
```

### 2. Start the Server

```bash
uv run uvicorn server.app:app --reload --port 8000
```

### 3. Run the Baseline Agent

```bash
export OPENAI_API_KEY="your-key-here"
uv run python baseline_agent.py --episodes 3
```

## Baseline Agent

The `baseline_agent.py` uses OpenAI's SDK supported models to solve tasks:

```bash
# Run with default settings (gpt-4o-mini, 3 episodes per difficulty)
uv run python baseline_agent.py

# Use a different model
uv run python baseline_agent.py --model gpt-4o

# Test specific difficulty
uv run python baseline_agent.py --difficulty medium --episodes 5

# Save results to JSON
uv run python baseline_agent.py --output results.json
```

### Expected Performance

| Difficulty | Expected Completion Rate | Expected Grade |
|------------|-------------------------|----------------|
| Easy | ~100% | ~1.00 |
| Medium | ~90%+ | ~0.90+ |
| Hard | ~70%+ | ~0.80+ |

## Environment API

### Action

```python
from models import ApiOpenAction

action = ApiOpenAction(
    api_name="get_user",
    args={"user_id": "U101"}
)
```

### Observation

```python
# After env.step(action):
obs.task_description  # "Generate invoice for user U101"
obs.available_apis    # ["get_user", "get_orders", ...]
obs.last_api_result   # {"success": True, "data": {...}}
obs.api_call_history  # List of all previous calls
obs.step_count        # Current step number
obs.task_complete     # Whether goal achieved
obs.done              # Whether episode ended
obs.reward            # Reward for this step
```

### Web Runner Args Format

When using the OpenEnv web runner UI, enter `Api Name` as plain text (for example, `get_ticket`) and enter `Args` as a JSON object string.

Example:

```json
{"ticket_id":"T302"}
```

The environment accepts this JSON text and parses it into the action `args` dictionary.

### Available Mock APIs

| API | Arguments | Returns |
|-----|-----------|---------|
| `get_user` | `user_id` | User info with email |
| `get_orders` | `user_id` | List of user's orders |
| `get_product` | `product_id` | Product details |
| `create_invoice` | `user_id`, `order_id` | Invoice confirmation |
| `get_ticket` | `ticket_id` | Support ticket details |
| `process_refund` | `user_id`, `order_id` | Refund status (fails if >30 days) |
| `send_email` | `email`, `subject`, `body` | Send confirmation |

## Reward Structure

The environment provides dense reward signals for learning:

| Event | Reward |
|-------|--------|
| Successful API call | +0.1 |
| Correct next API in sequence | +0.2 |
| Critical API (invoice, refund, email) | +0.15 to +0.2 |
| Failed API call | -0.1 |
| Invalid API (not available) | -0.3 |
| Per-step penalty | -0.02 |
| Task completion | +1.0 |

## Grading (0.0 - 1.0)

The `grade()` method scores episodes:

- **Task Completion**: 0.5 points
- **Efficiency** (fewer steps = better): 0.3 points
- **Sequence Correctness**: 0.2 points

## Direct Environment Usage

```python
from server.api_open_env_environment import ApiOpenEnvironment
from models import ApiOpenAction

env = ApiOpenEnvironment()

# Start episode
obs = env.reset(task_difficulty="medium")  # or "easy", "hard", "random"
print(f"Task: {obs.task_description}")

# Agent loop
while not obs.done:
    # Decide action based on observation
    action = ApiOpenAction(api_name="get_user", args={"user_id": "U101"})
    obs = env.step(action)
    print(f"Step {obs.step_count}: reward={obs.reward:.2f}")

# Final score
print(f"Grade: {env.grade():.2f}")
```

## Project Structure

```
api_open_env/
├── baseline_agent.py      # LLM-based agent using OpenAI API
├── test_env.py            # Test suite
├── models.py              # Action/Observation Pydantic models
├── client.py              # WebSocket client for remote server
├── openenv.yaml           # OpenEnv manifest with task definitions
├── pyproject.toml         # Dependencies including openai
└── server/
    ├── api_open_env_environment.py  # Core environment logic
    ├── mock_apis.py                 # Mock API implementations
    ├── mock_db.json                 # Sample data
    └── app.py                       # FastAPI server
```

## Requirements

- Python 3.10+
- OpenAI API key (for baseline agent)
- Dependencies: `uv sync`

## Running Tests

```bash
# Environment tests
uv run python test_env.py

# Validate OpenEnv compliance
uv run openenv validate
```

---

## 🤖 Training a Small Model (0.8B)

This project includes infrastructure to train a small Qwen 0.8B model using expert trajectories from larger models.

### Training Pipeline (5 Steps)

#### 1. **Collect Expert Trajectories**

```bash
# Set up Ollama with Qwen (or use OpenAI API)
export OPENAI_API_KEY="your-key"
export API_BASE_URL="http://localhost:11434/v1"  # for Ollama
export MODEL_NAME="qwen2.5:14b"

# Collect 100+ successful trajectories (grade >= 0.8)
uv run python collect_trajectories.py \
  --model qwen2.5:14b \
  --episodes 100 \
  --min-grade 0.8 \
  --output data/qwen_trajectories.jsonl
```

#### 2. **Prepare Training Data**

```bash
# Convert to fine-tuning format
uv run python prepare_training_data.py \
  --input data/qwen_trajectories.jsonl \
  --output data/training_data.jsonl \
  --format openai
```

#### 3. **Train with LoRA**

```bash
# Install training dependencies
pip install unsloth trl peft bitsandbytes

# Fine-tune Qwen 0.8B (takes 2-4 hours on single GPU)
uv run python train_model.py
```

The training script:
- Uses **4-bit quantization** for memory efficiency
- Applies **LoRA adapters** (rank=16) to key layers
- Trains for **3 epochs** with cosine LR schedule
- Saves to `./trained_model/`

#### 4. **Evaluate Performance**

```bash
# Compare base model vs trained model
uv run python baseline_agent.py --model qwen2.5:0.8b --episodes 10 --output results/base.json
uv run python baseline_agent.py --model ./trained_model --episodes 10 --output results/trained.json

# Generate comparison report
uv run python compare_models.py \
  results/base.json \
  results/trained.json \
  --names "Qwen 0.8B Base" "Qwen 0.8B Trained"
```

#### 5. **Visualize Results**

The comparison script generates:
- Performance tables by difficulty
- Completion rate comparisons  
- Grade improvement charts
- `model_comparison.png` graph

### Expected Training Results

| Model | Completion Rate | Avg Grade | Speed |
|-------|----------------|-----------|-------|
| **Qwen 14B (Expert)** | ~90% | 0.85 | 1x |
| **Qwen 0.8B (Base)** | ~40% | 0.45 | 10x |
| **Qwen 0.8B (Trained)** | **~65%** | **0.65** | 10x |

**Goal**: Achieve 60-70% of expert performance at 17.5x smaller model size.

### Training Configuration

- **Base Model**: Qwen/Qwen2.5-0.8B-Instruct
- **Method**: Supervised Fine-Tuning (SFT) with LoRA
- **Data**: 100-300 successful trajectories
- **LoRA Rank**: 16 (balances quality vs efficiency)
- **Batch Size**: 2 (with 4x gradient accumulation)
- **Learning Rate**: 2e-4
- **Hardware**: Single GPU with 8GB+ VRAM
- **Time**: 2-4 hours

### Files for Training

| File | Purpose |
|------|---------|
| `collect_trajectories.py` | Gather expert demonstrations |
| `prepare_training_data.py` | Convert to training format |
| `train_model.py` | LoRA fine-tuning script |
| `compare_models.py` | Evaluation & comparison |
| `baseline_agent.py` | Run inference with any model |

---

## 🚀 Hackathon Inference Script

For hackathon submission, use the standardized inference script:

```bash
# Set environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"

# Run inference on all tasks
uv run python inference.py
```

**Output Format** (required for hackathon):
```
[START] task=easy env=api-workflow-env model=gpt-4o-mini
[STEP] step=1 action=get_user({"user_id":"U101"}) reward=0.10 done=false error=null
[STEP] step=2 action=get_orders({"user_id":"U101"}) reward=0.20 done=false error=null
...
[END] success=true steps=5 score=0.850 rewards=0.10,0.20,0.30,0.15,1.00
```

The script:
- ✅ Runs all 3 tasks (easy, medium, hard)
- ✅ Uses OpenAI-compatible client
- ✅ Reads credentials from environment
- ✅ Outputs exact [START]/[STEP]/[END] format
- ✅ Completes within 20 minutes
- ✅ Returns scores in [0, 1] range

