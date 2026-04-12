import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))

from models import ApiOpenAction, ApiOpenObservation
from server.api_open_env_environment import ApiOpenEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN or OPENAI_API_KEY must be set")

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "api-workflow-env"
SCORE_FLOOR = 0.001
SCORE_CEILING = 0.999


def to_open_unit_interval(score: float) -> float:
    return min(SCORE_CEILING, max(SCORE_FLOOR, score))

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_user",
            "description": "Retrieve user information by user ID",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_orders",
            "description": "Retrieve all orders for a given user",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Retrieve product information by product ID",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_invoice",
            "description": "Create an invoice for a user's order",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "order_id": {"type": "string"},
                },
                "required": ["user_id", "order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket",
            "description": "Retrieve support ticket information",
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_refund",
            "description": "Process a refund for a user's order",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "order_id": {"type": "string"},
                },
                "required": ["user_id", "order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email notification to the specified address",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["email"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an AI agent that completes tasks by calling APIs in the correct sequence.
Use function/tool calling to select the next API and arguments.
Only call APIs from the available tool list for the current step."""


def call_model_with_retry(call_fn, retries: int = 5, base_wait: float = 1.0, max_wait: float = 30.0):
    for i in range(retries):
        try:
            return call_fn()
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                wait = min(base_wait * (2**i), max_wait)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Max retries exceeded")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def format_observation_for_llm(obs: ApiOpenObservation) -> str:
    lines = [
        f"TASK: {obs.task_description}",
        f"AVAILABLE APIs: {', '.join(obs.available_apis)}",
        f"STEP: {obs.step_count}/{obs.max_steps}",
    ]
    if obs.api_call_history:
        lines.append("\nPREVIOUS API CALLS:")
        for i, call in enumerate(obs.api_call_history, 1):
            result = call.get("result", {})
            if result.get("success", False):
                data_str = json.dumps(result.get("data", {}))
                if len(data_str) > 400:
                    data_str = data_str[:400] + "..."
                lines.append(f"  {i}. {call['api_name']}({call['args']}) -> SUCCESS")
                lines.append(f"     Data: {data_str}")
            else:
                lines.append(
                    f"  {i}. {call['api_name']}({call['args']}) -> FAILED: {result.get('error', 'Unknown error')}"
                )
    else:
        lines.append("\nNo API calls made yet. Start by gathering information.")
    return "\n".join(lines)


def parse_content_json(content: str) -> Optional[Dict[str, Any]]:
    content = content.strip()
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        content = content[start:end].strip()
    elif "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        content = content[start:end].strip()
    try:
        parsed = json.loads(content)
        if parsed.get("api_name"):
            return {"api_name": parsed["api_name"], "args": parsed.get("args", {})}
    except Exception:
        return None
    return None


def get_action_from_llm(client: OpenAI, obs: ApiOpenObservation) -> Optional[ApiOpenAction]:
    user_prompt = format_observation_for_llm(obs)
    available_tools = [t for t in TOOL_DEFINITIONS if t["function"]["name"] in obs.available_apis]

    def _call():
        return client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tools=available_tools,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=500,
        )

    try:
        response = call_model_with_retry(_call)
        message = response.choices[0].message

        if getattr(message, "tool_calls", None):
            call = message.tool_calls[0]
            args = json.loads(call.function.arguments or "{}")
            return ApiOpenAction(api_name=call.function.name, args=args)

        if message.content:
            parsed = parse_content_json(message.content)
            if parsed:
                return ApiOpenAction(api_name=parsed["api_name"], args=parsed["args"])
    except Exception:
        return None

    return None


def run_task(client: OpenAI, env: ApiOpenEnvironment, task_difficulty: str) -> Dict[str, Any]:
    log_start(task=task_difficulty, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = SCORE_FLOOR
    success = False

    try:
        obs = env.reset(task_difficulty=task_difficulty)
        while not obs.done:
            steps_taken += 1

            action = get_action_from_llm(client, obs)
            if action is None:
                log_step(
                    step=steps_taken,
                    action="no_action({})",
                    reward=0.0,
                    done=True,
                    error="Model returned no valid tool call",
                )
                break

            obs = env.step(action)
            reward = obs.reward or 0.0
            rewards.append(reward)

            error = None
            if obs.last_api_result and not obs.last_api_result.get("success", False):
                error = obs.last_api_result.get("error", "Unknown error")

            log_step(
                step=steps_taken,
                action=f"{action.api_name}({json.dumps(action.args)})",
                reward=reward,
                done=obs.done,
                error=error,
            )

            time.sleep(1.0)

        score = to_open_unit_interval(env.grade())
        success = obs.task_complete
    except Exception:
        success = False
        score = SCORE_FLOOR
    finally:
        score = to_open_unit_interval(score)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_difficulty, "success": success, "steps": steps_taken, "score": score, "rewards": rewards}


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = ApiOpenEnvironment()
    for task in TASKS:
        run_task(client, env, task)


if __name__ == "__main__":
    main()
