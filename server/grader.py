"""
Grader for evaluating agent performance on API Workflow tasks.

The grader assigns a score from 0.0 to 1.0 based on:
- Task completion (primary)
- Efficiency (minimal unnecessary API calls)
- Success rate (avoiding failed calls)
"""

from typing import Dict, List, Any


def grade_episode(
    task_config: Dict[str, Any],
    api_call_history: List[Dict[str, Any]],
    task_complete: bool,
) -> Dict[str, Any]:
    """
    Grade an episode based on task completion and efficiency.

    Args:
        task_config: Configuration of the task (difficulty, max_steps, etc.)
        api_call_history: List of all API calls made during the episode
        task_complete: Whether the task was successfully completed

    Returns:
        Dictionary with score (0.0-1.0) and breakdown
    """
    if not task_complete:
        # No credit for incomplete tasks
        return {
            "score": 0.0,
            "task_complete": False,
            "breakdown": {
                "completion": 0.0,
                "efficiency": 0.0,
                "success_rate": 0.0,
            },
            "feedback": "Task not completed",
        }

    # Task completed - calculate quality metrics
    total_calls = len(api_call_history)
    successful_calls = sum(1 for call in api_call_history if call.get("result", {}).get("success", False))
    failed_calls = total_calls - successful_calls

    # Calculate success rate (0.0 to 1.0)
    success_rate = successful_calls / total_calls if total_calls > 0 else 0.0

    # Calculate efficiency based on task difficulty
    difficulty = task_config.get("difficulty", "medium")
    optimal_steps = {"easy": 1, "medium": 4, "hard": 5}
    expected_steps = optimal_steps.get(difficulty, 4)

    # Efficiency score: penalize excessive API calls
    if total_calls <= expected_steps:
        efficiency = 1.0
    else:
        # Decay efficiency for extra calls
        extra_calls = total_calls - expected_steps
        efficiency = max(0.0, 1.0 - (extra_calls * 0.1))

    # Weighted scoring
    completion_weight = 0.5
    efficiency_weight = 0.3
    success_rate_weight = 0.2

    score = (
        (1.0 * completion_weight) + (efficiency * efficiency_weight) + (success_rate * success_rate_weight)
    )

    # Ensure score is in [0.0, 1.0]
    score = max(0.0, min(1.0, score))

    feedback = []
    if score >= 0.9:
        feedback.append("Excellent! Task completed efficiently.")
    elif score >= 0.7:
        feedback.append("Good! Task completed with minor inefficiencies.")
    else:
        feedback.append("Task completed but with significant inefficiencies.")

    if failed_calls > 0:
        feedback.append(f"Made {failed_calls} failed API call(s).")

    if total_calls > expected_steps + 2:
        feedback.append(f"Used {total_calls - expected_steps} more API calls than optimal.")

    return {
        "score": round(score, 3),
        "task_complete": True,
        "breakdown": {
            "completion": 1.0,
            "efficiency": round(efficiency, 3),
            "success_rate": round(success_rate, 3),
        },
        "metrics": {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "expected_steps": expected_steps,
        },
        "feedback": " ".join(feedback),
    }


def grade_task_easy(api_call_history: List[Dict[str, Any]]) -> float:
    """
    Grade Task 1 (Easy): Fetch user email.

    Perfect score: Call get_user once successfully.

    Args:
        api_call_history: API call history

    Returns:
        Score from 0.0 to 1.0
    """
    if not api_call_history:
        return 0.0

    # Check if get_user was called successfully
    for call in api_call_history:
        if call["api_name"] == "get_user" and call.get("result", {}).get("success"):
            # Penalize for additional calls
            if len(api_call_history) == 1:
                return 1.0
            elif len(api_call_history) == 2:
                return 0.9
            else:
                return max(0.6, 1.0 - (len(api_call_history) - 1) * 0.1)

    return 0.0


def grade_task_medium(api_call_history: List[Dict[str, Any]]) -> float:
    """
    Grade Task 2 (Medium): Generate invoice.

    Perfect score: get_user → get_orders → get_product → create_invoice (4 steps).

    Args:
        api_call_history: API call history

    Returns:
        Score from 0.0 to 1.0
    """
    if not api_call_history:
        return 0.0

    # Check if invoice was created
    invoice_created = any(
        call["api_name"] == "create_invoice" and call.get("result", {}).get("success")
        for call in api_call_history
    )

    if not invoice_created:
        return 0.0

    # Perfect path: 4 steps
    total_calls = len(api_call_history)
    if total_calls == 4:
        return 1.0
    elif total_calls <= 5:
        return 0.9
    elif total_calls <= 6:
        return 0.8
    else:
        return max(0.6, 1.0 - (total_calls - 4) * 0.1)


def grade_task_hard(api_call_history: List[Dict[str, Any]]) -> float:
    """
    Grade Task 3 (Hard): Resolve support ticket with refund.

    Perfect score: get_ticket → get_user → get_orders → process_refund → send_email (5 steps).

    Args:
        api_call_history: API call history

    Returns:
        Score from 0.0 to 1.0
    """
    if not api_call_history:
        return 0.0

    # Check if refund was processed and email sent
    refund_processed = any(
        call["api_name"] == "process_refund" and call.get("result", {}).get("success")
        for call in api_call_history
    )
    email_sent = any(
        call["api_name"] == "send_email" and call.get("result", {}).get("success")
        for call in api_call_history
    )

    if not (refund_processed and email_sent):
        # Partial credit if only refund or only email
        if refund_processed:
            return 0.4
        elif email_sent:
            return 0.2
        return 0.0

    # Both required actions completed
    total_calls = len(api_call_history)
    if total_calls == 5:
        return 1.0
    elif total_calls <= 6:
        return 0.95
    elif total_calls <= 7:
        return 0.85
    else:
        return max(0.7, 1.0 - (total_calls - 5) * 0.05)
