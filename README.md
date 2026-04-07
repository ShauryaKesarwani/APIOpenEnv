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

The `baseline_agent.py` uses OpenAI's GPT models to solve tasks:

```bash
# Run with default settings (gpt-4o-mini, 3 episodes per difficulty)
python baseline_agent.py

# Use a different model
python baseline_agent.py --model gpt-4o

# Test specific difficulty
python baseline_agent.py --difficulty medium --episodes 5

# Save results to JSON
python baseline_agent.py --output results.json
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
