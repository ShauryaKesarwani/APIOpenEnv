"""
OpenEnv Hackathon Inference Script

MANDATORY compliance with hackathon requirements:
- Uses OpenAI-compatible client
- Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
- Outputs [START]/[STEP]/[END] format exactly
- Runs all 3 tasks (easy, medium, hard)
- Completes within 20 minutes
"""

import os
import sys
import json
import time
from typing import List, Optional

from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from server.api_open_env_environment import ApiOpenEnvironment
from models import ApiOpenAction

# Environment variables (MANDATORY for hackathon)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN or OPENAI_API_KEY must be set")

# Task configuration
TASKS = ["easy", "medium", "hard"]
BENCHMARK = "api-workflow-env"
MAX_STEPS_BY_DIFFICULTY = {
    "easy": 3,
    "medium": 8,
    "hard": 10
}

SYSTEM_PROMPT = """You are an AI agent that completes tasks by calling APIs in the correct sequence.

You will receive:
- A task description
- Available APIs you can call
- Results from previous API calls (if any)

Your job is to decide which API to call next and with what arguments.

Respond with a JSON object containing:
{
    "api_name": "name_of_api_to_call",
    "args": {"arg1": "value1", ...}
}

API Signatures:
- get_user(user_id: str) -> Returns user info including email
- get_orders(user_id: str) -> Returns list of orders
- get_product(product_id: str) -> Returns product info
- create_invoice(user_id: str, order_id: str) -> Creates invoice
- get_ticket(ticket_id: str) -> Returns ticket info
- process_refund(user_id: str, order_id: str) -> Processes refund (only if order within 30 days)
- send_email(email: str, subject: str, body: str) -> Sends email
"""


def log_start(task: str, env: str, model: str) -> None:
    """[START] log line"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """[STEP] log line"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """[END] log line"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


def format_observation(obs) -> str:
    """Format observation for LLM prompt"""
    lines = [
        f"TASK: {obs.task_description}",
        f"AVAILABLE APIs: {', '.join(obs.available_apis)}",
        f"STEP: {obs.step_count + 1}/{obs.max_steps}",
    ]
    
    if obs.api_call_history:
        lines.append("\nPREVIOUS API CALLS:")
        for i, call in enumerate(obs.api_call_history, 1):
            result = call.get("result", {})
            success = result.get("success", False)
            if success:
                data = result.get("data", {})
                data_str = json.dumps(data)
                if len(data_str) > 300:
                    data_str = data_str[:300] + "..."
                lines.append(f"  {i}. {call['api_name']}({call['args']}) -> SUCCESS")
                lines.append(f"     Data: {data_str}")
            else:
                error = result.get("error", "Unknown error")
                lines.append(f"  {i}. {call['api_name']}({call['args']}) -> FAILED: {error}")
    else:
        lines.append("\nNo API calls made yet. Start by gathering information.")
    
    lines.append("\nRespond with JSON containing api_name and args.")
    return "\n".join(lines)


def get_action_from_llm(client: OpenAI, obs, max_retries: int = 2) -> Optional[ApiOpenAction]:
    """Get action from LLM with retry logic"""
    user_prompt = format_observation(obs)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            
            llm_response = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if "```json" in llm_response:
                start = llm_response.find("```json") + 7
                end = llm_response.find("```", start)
                llm_response = llm_response[start:end].strip()
            elif "```" in llm_response:
                start = llm_response.find("```") + 3
                end = llm_response.find("```", start)
                llm_response = llm_response[start:end].strip()
            
            # Parse JSON
            data = json.loads(llm_response)
            
            if "api_name" in data:
                return ApiOpenAction(
                    api_name=data["api_name"],
                    args=data.get("args", {})
                )
        
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[DEBUG] Failed to parse LLM response after {max_retries} attempts: {e}", flush=True)
                return None
            time.sleep(0.5)
    
    return None


def run_task(client: OpenAI, env: ApiOpenEnvironment, task_difficulty: str) -> dict:
    """Run a single task and return results"""
    max_steps = MAX_STEPS_BY_DIFFICULTY.get(task_difficulty, 10)
    
    log_start(task=task_difficulty, env=BENCHMARK, model=MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        # Reset environment
        obs = env.reset(task_difficulty=task_difficulty)
        
        # Run episode
        while not obs.done and steps_taken < max_steps:
            steps_taken += 1
            
            # Get action from LLM
            action = get_action_from_llm(client, obs)
            
            if not action:
                # Fallback: try first available API
                action = ApiOpenAction(
                    api_name=obs.available_apis[0] if obs.available_apis else "get_user",
                    args={}
                )
            
            # Execute action
            obs = env.step(action)
            
            reward = obs.reward or 0.0
            rewards.append(reward)
            
            # Format action for logging
            action_str = f"{action.api_name}({json.dumps(action.args)})"
            
            # Get error if any
            error = None
            if obs.last_api_result and not obs.last_api_result.get("success", False):
                error = obs.last_api_result.get("error", "Unknown error")
            
            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=obs.done,
                error=error
            )
            
            # Add delay to prevent overload
            time.sleep(1.0)
        
        # Calculate final score (normalized to [0, 1])
        score = env.grade()
        success = obs.task_complete
        
    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
        success = False
        score = 0.0
    
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return {
        "task": task_difficulty,
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards
    }


def main():
    """Main inference script - runs all tasks"""
    print(f"[INFO] Starting inference with model: {MODEL_NAME}", flush=True)
    print(f"[INFO] API endpoint: {API_BASE_URL}", flush=True)
    print(f"[INFO] Running tasks: {TASKS}", flush=True)
    print("=" * 60, flush=True)
    
    # Initialize client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Initialize environment
    env = ApiOpenEnvironment()
    
    # Run all tasks
    all_results = []
    for task in TASKS:
        print(f"\n[INFO] Running task: {task}", flush=True)
        result = run_task(client, env, task)
        all_results.append(result)
        print(f"[INFO] Task {task} completed: score={result['score']:.3f}", flush=True)
    
    # Summary
    print("\n" + "=" * 60, flush=True)
    print("[INFO] INFERENCE COMPLETE", flush=True)
    print("=" * 60, flush=True)
    
    for result in all_results:
        print(
            f"  {result['task']:8s}: score={result['score']:.3f}, "
            f"success={result['success']}, steps={result['steps']}",
            flush=True
        )
    
    avg_score = sum(r['score'] for r in all_results) / len(all_results)
    print(f"\nAverage Score: {avg_score:.3f}", flush=True)
    
    return all_results


if __name__ == "__main__":
    main()
