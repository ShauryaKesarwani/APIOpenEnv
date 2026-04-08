"""
Convert collected trajectories to training format for fine-tuning.
Supports both OpenAI fine-tuning format and HuggingFace format.
"""

import os
import json
import argparse
from typing import List, Dict, Any


def trajectory_to_openai_format(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a single trajectory to OpenAI fine-tuning format with tool calls.
    
    Returns:
        List of training examples (one per step in trajectory)
    """
    examples = []
    
    system_message = {
        "role": "system",
        "content": """You are an AI agent that completes tasks by calling APIs in the correct sequence.

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
5. For ticket tasks: get_ticket -> get_user -> get_orders -> process_refund (if eligible) -> send_email"""
    }
    
    # Build conversation for each step
    for i, step in enumerate(trajectory["steps"]):
        messages = [system_message]
        
        # Build user prompt with current state
        obs = step["observation"]
        user_content_lines = [
            f"TASK: {obs['task']}",
            f"AVAILABLE APIs: {', '.join(obs['available_apis'])}",
            f"STEP: {step['step_num'] + 1}/10",
        ]
        
        # Add previous API call history
        if obs.get('api_call_history'):
            user_content_lines.append("\nPREVIOUS API CALLS:")
            for j, call in enumerate(obs['api_call_history'], 1):
                result = call.get("result", {})
                success = result.get("success", False)
                if success:
                    data = result.get("data", {})
                    data_str = json.dumps(data, indent=2)
                    if len(data_str) > 300:
                        data_str = data_str[:300] + "..."
                    user_content_lines.append(f"  {j}. {call['api_name']}({call['args']}) -> SUCCESS")
                    user_content_lines.append(f"     Data: {data_str}")
                else:
                    error = result.get("error", "Unknown error")
                    user_content_lines.append(f"  {j}. {call['api_name']}({call['args']}) -> FAILED: {error}")
        else:
            user_content_lines.append("\nNo API calls made yet. Start by gathering information.")
        
        messages.append({
            "role": "user",
            "content": "\n".join(user_content_lines)
        })
        
        # Add assistant response with tool call
        action = step["action"]
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": action["api_name"],
                    "arguments": json.dumps(action["args"])
                }
            }]
        })
        
        # Add tool response
        result = step["result"]
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": json.dumps(result)
        })
        
        examples.append({"messages": messages})
    
    return examples


def trajectory_to_text_format(trajectory: Dict[str, Any]) -> str:
    """
    Convert trajectory to simple text format for instruction fine-tuning.
    Useful for models that don't support tool calling.
    """
    text_parts = []
    
    text_parts.append("### Task")
    text_parts.append(trajectory["metadata"]["task"])
    text_parts.append("")
    
    text_parts.append("### Available APIs")
    text_parts.append(", ".join(trajectory["metadata"]["available_apis"]))
    text_parts.append("")
    
    text_parts.append("### Solution")
    for i, step in enumerate(trajectory["steps"], 1):
        action = step["action"]
        result = step["result"]
        success = result.get("success", False)
        
        text_parts.append(f"Step {i}: {action['api_name']}({json.dumps(action['args'])})")
        if success:
            text_parts.append(f"Result: SUCCESS - {json.dumps(result.get('data', {}))}")
        else:
            text_parts.append(f"Result: FAILED - {result.get('error', 'Unknown error')}")
        text_parts.append("")
    
    return "\n".join(text_parts)


def prepare_dataset(
    input_file: str,
    output_file: str,
    format_type: str = "openai",
    max_examples: int = None
) -> Dict[str, Any]:
    """
    Convert trajectories file to training format.
    
    Args:
        input_file: Input trajectories file (.jsonl)
        output_file: Output training file
        format_type: "openai" or "text"
        max_examples: Maximum examples to include (None = all)
        
    Returns:
        Statistics about the dataset
    """
    print("=" * 60)
    print("PREPARING TRAINING DATASET")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Format: {format_type}")
    print()
    
    # Load trajectories
    trajectories = []
    with open(input_file) as f:
        for line in f:
            trajectories.append(json.loads(line))
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Convert to training format
    training_examples = []
    
    for i, traj in enumerate(trajectories):
        if max_examples and i >= max_examples:
            break
            
        if format_type == "openai":
            examples = trajectory_to_openai_format(traj)
            training_examples.extend(examples)
        elif format_type == "text":
            text = trajectory_to_text_format(traj)
            training_examples.append({"text": text})
        else:
            raise ValueError(f"Unknown format: {format_type}")
    
    print(f"Generated {len(training_examples)} training examples")
    
    # Save training data
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    # Calculate statistics
    stats = {
        "num_trajectories": len(trajectories),
        "num_examples": len(training_examples),
        "format": format_type,
        "avg_examples_per_trajectory": len(training_examples) / len(trajectories) if trajectories else 0
    }
    
    # Save stats
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Training examples: {stats['num_examples']}")
    print(f"Avg examples per trajectory: {stats['avg_examples_per_trajectory']:.1f}")
    print(f"\nSaved to: {output_file}")
    print(f"Stats saved to: {stats_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from trajectories")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input trajectories file (.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output training file (.jsonl)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="openai",
        choices=["openai", "text"],
        help="Output format (openai for tool calling, text for instruction tuning)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of trajectories to convert"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        input_file=args.input,
        output_file=args.output,
        format_type=args.format,
        max_examples=args.max_examples
    )


if __name__ == "__main__":
    main()
