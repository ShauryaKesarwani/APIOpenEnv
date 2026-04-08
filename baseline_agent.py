"""
Baseline Inference Script for API Workflow Environment.

Uses OpenAI API to run an LLM-based agent against the environment.
The agent observes the task, available APIs, and previous results,
then decides which API to call next.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python baseline_agent.py

    # Or with specific settings:
    python baseline_agent.py --model gpt-4o-mini --episodes 5 --difficulty all
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import time

load_dotenv(dotenv_path=".env")

def call_model_with_retry(call_fn, retries=5, base_wait=1, max_wait=30):
    for i in range(retries):
        try:
            return call_fn()
        except Exception as e:
            error_msg = str(e).lower()

            if "429" in error_msg or "rate limit" in error_msg:
                wait = min(base_wait * (2 ** i), max_wait)
                print(f"Rate limited. Retrying in {wait}s... ({i+1}/{retries})")
                time.sleep(wait)
            else:
                raise

    raise Exception("Max retries exceeded")


sys.path.insert(0, ".")

from openai import OpenAI
from server.api_open_env_environment import ApiOpenEnvironment
from models import ApiOpenAction, ApiOpenObservation


# Tool definitions for OpenAI-style function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_user",
            "description": "Retrieve user information by user ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID to look up (e.g., U101, U102)"
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_orders",
            "description": "Retrieve all orders for a given user",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID to look up orders for"
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Retrieve product information by product ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID to look up (e.g., P701, P702)"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_invoice",
            "description": "Create an invoice for a user's order",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID"
                    },
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to create invoice for"
                    }
                },
                "required": ["user_id", "order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket",
            "description": "Retrieve support ticket information",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The ticket ID to look up (e.g., T301, T302)"
                    }
                },
                "required": ["ticket_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_refund",
            "description": "Process a refund for a user's order. Refunds are only valid if the order is within 30 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID"
                    },
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to refund"
                    }
                },
                "required": ["user_id", "order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email notification to the specified address",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The recipient email address"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content"
                    }
                },
                "required": ["email"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are an AI agent that completes tasks by calling APIs in the correct sequence.

You will receive:
- A task description
- Available APIs you can call via function calling
- Results from previous API calls (if any)

Your job is to decide which API to call next using function calling.

IMPORTANT RULES:
1. Only use APIs from the available tools
2. Use information from previous API results to inform your next action
3. Think step-by-step about what information you need
4. For invoice tasks: get_user -> get_orders -> get_product -> create_invoice
5. For ticket tasks: get_ticket -> get_user -> get_orders -> process_refund (if eligible) -> send_email
"""


def format_observation_for_llm(obs: ApiOpenObservation) -> str:
    """Format the observation as a prompt for the LLM."""
    lines = [
        f"TASK: {obs.task_description}",
        f"AVAILABLE APIs: {', '.join(obs.available_apis)}",
        f"STEP: {obs.step_count}/{obs.max_steps}",
    ]

    if obs.api_call_history:
        lines.append("\nPREVIOUS API CALLS:")
        for i, call in enumerate(obs.api_call_history, 1):
            result = call.get("result", {})
            success = result.get("success", False)
            if success:
                data = result.get("data", {})
                # Truncate long data for readability
                data_str = json.dumps(data, indent=2)
                if len(data_str) > 500:
                    data_str = data_str[:500] + "..."
                lines.append(f"  {i}. {call['api_name']}({call['args']}) -> SUCCESS")
                lines.append(f"     Data: {data_str}")
            else:
                error = result.get("error", "Unknown error")
                lines.append(f"  {i}. {call['api_name']}({call['args']}) -> FAILED: {error}")
    else:
        lines.append("\nNo API calls made yet. Start by gathering information.")

    lines.append("\nWhat API should be called next? Respond with JSON.")
    return "\n".join(lines)


def parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse the LLM response to extract API call."""
    try:
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        # Parse JSON
        data = json.loads(response)
        return {
            "api_name": data.get("api_name"),
            "args": data.get("args", {}),
            "reasoning": data.get("reasoning", ""),
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [Warning] Failed to parse LLM response: {e}")
        return None


class BaselineAgent:
    """LLM-based agent for the API Workflow Environment."""

    def __init__(self, model: str = "gpt-4o-mini", verbose: bool = True, use_tools: bool = True):
        """
        Initialize the agent with OpenAI client.

        Args:
            model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
            verbose: Whether to print agent's reasoning
            use_tools: Whether to use tool calling format (True) or legacy JSON format (False)
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        inference_server = os.environ.get("INFERENCE_SERVER")
        model_name = os.environ.get("MODEL_UPPER_NAME")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it with: export OPENAI_API_KEY='your-key-here'"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.verbose = verbose
        self.use_tools = use_tools

    def get_action(self, obs: ApiOpenObservation) -> Optional[ApiOpenAction]:
        """
        Use LLM to decide the next action based on observation.

        Args:
            obs: Current observation from environment

        Returns:
            ApiOpenAction to execute, or None if parsing failed
        """
        user_prompt = format_observation_for_llm(obs)

        try:
            # Filter tools to only include available APIs for this task
            available_tools = [
                tool for tool in TOOL_DEFINITIONS 
                if tool["function"]["name"] in obs.available_apis
            ] if self.use_tools else None

            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,  # Low temperature for consistency
                "max_tokens": 500,
            }
            
            # Add tools if using tool calling format
            if self.use_tools and available_tools:
                request_params["tools"] = available_tools
                request_params["tool_choice"] = "auto"
            
            response = self.client.chat.completions.create(**request_params)
            message = response.choices[0].message

            # Handle tool calls (new format)
            if self.use_tools and hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if self.verbose:
                    print(f"  [Agent] Tool Call: {function_name}")
                    print(f"  [Agent] Arguments: {function_args}")
                
                return ApiOpenAction(
                    api_name=function_name,
                    args=function_args,
                )
            
            # Handle legacy JSON response format
            elif message.content:
                llm_response = message.content
                parsed = parse_llm_response(llm_response)

                if parsed and parsed.get("api_name"):
                    if self.verbose:
                        print(f"  [Agent] Reasoning: {parsed.get('reasoning', 'N/A')}")
                        print(f"  [Agent] Action: {parsed['api_name']}({parsed['args']})")

                    return ApiOpenAction(
                        api_name=parsed["api_name"],
                        args=parsed["args"],
                    )
                else:
                    print(f"  [Agent] Could not parse response: {llm_response[:200]}")
                    return None
            else:
                print(f"  [Agent] No tool call or content in response")
                return None

        except Exception as e:
            print(f"  [Agent] API error: {e}")
            return None


def run_episode(
    agent: BaselineAgent,
    env: ApiOpenEnvironment,
    difficulty: str,
    episode_num: int = 1,
) -> Dict[str, Any]:
    """
    Run a single episode with the agent.

    Returns:
        Dictionary with episode results
    """
    print(f"\n{'='*60}")
    print(f"Episode {episode_num} - Difficulty: {difficulty.upper()}")
    print("=" * 60)

    obs = env.reset(task_difficulty=difficulty)
    print(f"Task: {obs.task_description}")
    print(f"Available APIs: {obs.available_apis}")

    total_reward = 0.0
    steps = 0
    max_retries = 2  # Allow retries for parsing failures

    while not obs.done:
        print(f"\n--- Step {obs.step_count + 1}/{obs.max_steps} ---")

        # Get action from agent
        action = None
        for retry in range(max_retries):
            action = agent.get_action(obs)
            if action:
                break
            print(f"  [Retry {retry + 1}/{max_retries}]")

        if not action:
            # Fallback: try a safe default action
            print("  [Fallback] Using default action")
            if "get_user" in obs.available_apis and obs.step_count == 0:
                # Extract user_id from task if possible
                task = obs.task_description
                for uid in ["U101", "U102", "U103", "U104", "U105"]:
                    if uid in task:
                        action = ApiOpenAction(api_name="get_user", args={"user_id": uid})
                        break
            if not action:
                action = ApiOpenAction(api_name=obs.available_apis[0], args={})

        # Execute action
        obs = env.step(action)
        total_reward += obs.reward
        steps += 1

        # Print result
        success = obs.last_api_result.get("success", False) if obs.last_api_result else False
        print(f"  [Result] Success: {success}, Reward: {obs.reward:.2f}")

        if obs.task_complete:
            print("\n*** TASK COMPLETED! ***")
        
        # Add 1 second delay to prevent overload
        time.sleep(1.0)

    # Get final grade
    grade = env.grade()

    result = {
        "difficulty": difficulty,
        "task": obs.task_description,
        "completed": obs.task_complete,
        "steps": steps,
        "max_steps": obs.max_steps,
        "total_reward": total_reward,
        "grade": grade,
    }

    print(f"\n--- Episode Summary ---")
    print(f"Completed: {obs.task_complete}")
    print(f"Steps: {steps}/{obs.max_steps}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Grade: {grade:.2f}")

    return result


def evaluate_agent(
    model: str = "gpt-4o-mini",
    episodes_per_difficulty: int = 3,
    difficulties: List[str] = None,
    verbose: bool = True,
    use_tools: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate the agent across multiple episodes and difficulties.

    Args:
        model: OpenAI model to use
        episodes_per_difficulty: Number of episodes per difficulty level
        difficulties: List of difficulties to test (default: all)
        verbose: Print detailed output
        use_tools: Whether to use tool calling format

    Returns:
        Evaluation results with scores
    """
    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]

    print("=" * 60)
    print("API WORKFLOW ENVIRONMENT - BASELINE AGENT EVALUATION")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tool Calling Mode: {use_tools}")
    print(f"Episodes per difficulty: {episodes_per_difficulty}")
    print(f"Difficulties: {difficulties}")

    agent = BaselineAgent(model=model, verbose=verbose, use_tools=use_tools)
    env = ApiOpenEnvironment()

    all_results = []

    for difficulty in difficulties:
        for ep in range(1, episodes_per_difficulty + 1):
            result = run_episode(agent, env, difficulty, ep)
            all_results.append(result)

    # Compute statistics
    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)

    stats_by_difficulty = {}
    for diff in difficulties:
        diff_results = [r for r in all_results if r["difficulty"] == diff]
        if diff_results:
            avg_grade = sum(r["grade"] for r in diff_results) / len(diff_results)
            completion_rate = sum(1 for r in diff_results if r["completed"]) / len(diff_results)
            avg_steps = sum(r["steps"] for r in diff_results) / len(diff_results)

            stats_by_difficulty[diff] = {
                "avg_grade": avg_grade,
                "completion_rate": completion_rate,
                "avg_steps": avg_steps,
                "episodes": len(diff_results),
            }

            print(f"\n{diff.upper()}:")
            print(f"  Completion Rate: {completion_rate:.1%}")
            print(f"  Average Grade: {avg_grade:.2f}")
            print(f"  Average Steps: {avg_steps:.1f}")

    # Overall stats
    overall_grade = sum(r["grade"] for r in all_results) / len(all_results)
    overall_completion = sum(1 for r in all_results if r["completed"]) / len(all_results)

    print(f"\nOVERALL:")
    print(f"  Completion Rate: {overall_completion:.1%}")
    print(f"  Average Grade: {overall_grade:.2f}")

    return {
        "model": model,
        "episodes": all_results,
        "stats_by_difficulty": stats_by_difficulty,
        "overall_grade": overall_grade,
        "overall_completion_rate": overall_completion,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline LLM agent on API Workflow Environment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Episodes per difficulty level (default: 3)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="all",
        choices=["easy", "medium", "hard", "all"],
        help="Difficulty level to test (default: all)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Use legacy JSON format instead of tool calling",
    )

    args = parser.parse_args()

    # Set difficulties
    if args.difficulty == "all":
        difficulties = ["easy", "medium", "hard"]
    else:
        difficulties = [args.difficulty]

    # Run evaluation
    results = evaluate_agent(
        model=args.model,
        episodes_per_difficulty=args.episodes,
        difficulties=difficulties,
        verbose=not args.quiet,
        use_tools=not args.no_tools,
    )

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            # Convert to JSON-serializable format
            json_results = {
                "model": results["model"],
                "overall_grade": results["overall_grade"],
                "overall_completion_rate": results["overall_completion_rate"],
                "stats_by_difficulty": results["stats_by_difficulty"],
                "episodes": results["episodes"],
            }
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
