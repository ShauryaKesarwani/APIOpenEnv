"""
Collect training trajectories from successful agent runs.
Saves detailed execution traces for fine-tuning.
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from baseline_agent import BaselineAgent
from server.api_open_env_environment import ApiOpenEnvironment
import time

load_dotenv(dotenv_path=".env")


def collect_trajectory(
    agent: BaselineAgent,
    env: ApiOpenEnvironment,
    difficulty: str,
    max_steps: int = 10
) -> Dict[str, Any]:
    """
    Collect a single trajectory with detailed information.
    
    Returns:
        Trajectory dict with states, actions, rewards, and metadata
    """
    obs = env.reset(task_difficulty=difficulty)
    
    trajectory = {
        "metadata": {
            "task": obs.task_description,
            "difficulty": difficulty,
            "available_apis": obs.available_apis,
            "timestamp": datetime.now().isoformat(),
        },
        "steps": [],
        "final_result": {}
    }
    
    while not obs.done:
        # Get action
        action = agent.get_action(obs)
        
        if not action:
            trajectory["final_result"]["failed_reason"] = "Agent failed to produce action"
            break
        
        # Record state before action
        step_data = {
            "step_num": obs.step_count,
            "observation": {
                "task": obs.task_description,
                "available_apis": obs.available_apis,
                "api_call_history": obs.api_call_history,
            },
            "action": {
                "api_name": action.api_name,
                "args": action.args,
            }
        }
        
        # Execute action
        obs = env.step(action)
        
        # Record result
        step_data["result"] = obs.last_api_result
        step_data["reward"] = obs.reward
        step_data["task_complete"] = obs.task_complete
        
        trajectory["steps"].append(step_data)
        
        # Add delay to prevent overload
        time.sleep(1.0)
    
    # Get final grade
    grade = env.grade()
    trajectory["final_result"] = {
        "completed": obs.task_complete,
        "grade": grade,
        "total_steps": obs.step_count,
        "max_steps": obs.max_steps,
        "total_reward": sum(step["reward"] for step in trajectory["steps"]),
    }
    
    return trajectory


def collect_dataset(
    model: str,
    num_episodes: int = 100,
    difficulties: List[str] = None,
    min_grade: float = 0.8,
    output_file: str = "data/trajectories.jsonl",
    use_tools: bool = True
) -> Dict[str, Any]:
    """
    Collect multiple trajectories for training dataset.
    
    Args:
        model: Model name or path
        num_episodes: Number of episodes to collect
        difficulties: List of difficulty levels
        min_grade: Minimum grade to keep trajectory (for filtering)
        output_file: Output file path
        use_tools: Whether to use tool calling format
        
    Returns:
        Statistics about collected data
    """
    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]
    
    print("=" * 60)
    print("TRAJECTORY COLLECTION")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Episodes: {num_episodes}")
    print(f"Difficulties: {difficulties}")
    print(f"Minimum grade filter: {min_grade}")
    print(f"Tool calling: {use_tools}")
    print()
    
    agent = BaselineAgent(model=model, verbose=False, use_tools=use_tools)
    env = ApiOpenEnvironment()
    
    all_trajectories = []
    stats = {
        "total_collected": 0,
        "total_kept": 0,
        "by_difficulty": {d: {"collected": 0, "kept": 0} for d in difficulties},
        "avg_grade": 0.0,
        "avg_steps": 0.0,
    }
    
    episodes_per_difficulty = num_episodes // len(difficulties)
    
    for difficulty in difficulties:
        print(f"\nCollecting {difficulty.upper()} episodes...")
        
        for ep in range(episodes_per_difficulty):
            print(f"  Episode {ep + 1}/{episodes_per_difficulty}...", end=" ")
            
            try:
                trajectory = collect_trajectory(agent, env, difficulty)
                stats["total_collected"] += 1
                stats["by_difficulty"][difficulty]["collected"] += 1
                
                # Filter by grade
                grade = trajectory["final_result"]["grade"]
                if grade >= min_grade:
                    all_trajectories.append(trajectory)
                    stats["total_kept"] += 1
                    stats["by_difficulty"][difficulty]["kept"] += 1
                    print(f"✓ (grade: {grade:.2f})")
                else:
                    print(f"✗ (grade: {grade:.2f}, below threshold)")
                    
            except Exception as e:
                print(f"ERROR: {e}")
                continue
    
    # Calculate statistics
    if all_trajectories:
        stats["avg_grade"] = sum(t["final_result"]["grade"] for t in all_trajectories) / len(all_trajectories)
        stats["avg_steps"] = sum(t["final_result"]["total_steps"] for t in all_trajectories) / len(all_trajectories)
    
    # Save trajectories
    print(f"\n{'=' * 60}")
    print(f"Saving {len(all_trajectories)} trajectories to {output_file}")

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        for traj in all_trajectories:
            f.write(json.dumps(traj) + '\n')
    
    # Save statistics
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total collected: {stats['total_collected']}")
    print(f"Total kept (grade >= {min_grade}): {stats['total_kept']}")
    print(f"Average grade: {stats['avg_grade']:.3f}")
    print(f"Average steps: {stats['avg_steps']:.1f}")
    print()
    
    for diff in difficulties:
        d_stats = stats["by_difficulty"][diff]
        print(f"{diff.upper()}: {d_stats['kept']}/{d_stats['collected']} kept")
    
    print(f"\nTrajectories saved to: {output_file}")
    print(f"Statistics saved to: {stats_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Collect training trajectories")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--min-grade", type=float, default=0.8, help="Minimum grade to keep")
    parser.add_argument("--output", type=str, default="data/trajectories.jsonl", help="Output file")
    parser.add_argument("--no-tools", action="store_true", help="Use JSON format instead of tools")
    parser.add_argument(
        "--difficulty",
        type=str,
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=["easy", "medium", "hard"],
        help="Difficulty levels to collect"
    )
    
    args = parser.parse_args()
    
    collect_dataset(
        model=args.model,
        num_episodes=args.episodes,
        difficulties=args.difficulty,
        min_grade=args.min_grade,
        output_file=args.output,
        use_tools=not args.no_tools
    )


if __name__ == "__main__":
    main()
