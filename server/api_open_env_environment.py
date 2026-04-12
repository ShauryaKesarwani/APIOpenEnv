"""
API Workflow Environment Implementation.

An environment where AI agents learn to orchestrate multi-step API workflows
to complete real-world tasks like generating invoices and processing refunds.
"""

import random
from uuid import uuid4
from typing import Dict, List, Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ApiOpenAction, ApiOpenObservation
    from .mock_apis import call_api, reset_mock_db
except ImportError:
    from models import ApiOpenAction, ApiOpenObservation
    from server.mock_apis import call_api, reset_mock_db


# Expected API sequences for grading
EXPECTED_SEQUENCES = {
    "easy": ["get_user"],
    "medium": ["get_user", "get_orders", "get_product", "create_invoice"],
    "hard_refund": ["get_ticket", "get_user", "get_orders", "process_refund", "send_email"],
    "hard_no_refund": ["get_ticket", "get_user", "get_orders", "send_email"],
}

SCORE_FLOOR = 0.001
SCORE_CEILING = 0.999


def _to_open_unit_interval(score: float) -> float:
    """Map a score to the strict open interval (0, 1)."""
    return min(SCORE_CEILING, max(SCORE_FLOOR, score))


def _check_hard_success(history: List[Dict[str, Any]], ticket_id: str) -> bool:
    """Check success for hard task based on ticket type."""
    email_sent = any(
        call["api_name"] == "send_email" and call.get("result", {}).get("success")
        for call in history
    )

    if ticket_id == "T301":
        # T301 is within 30 days, refund should succeed
        refund_processed = any(
            call["api_name"] == "process_refund" and call.get("result", {}).get("success")
            for call in history
        )
        return refund_processed and email_sent
    else:
        # T302 is >30 days, refund should be denied, but email required
        refund_attempted = any(call["api_name"] == "process_refund" for call in history)
        return email_sent and refund_attempted


# Task definitions with success criteria
TASKS = {
    "easy": {
        "task_id": "TASK_EASY",
        "description": "Fetch user email for user {user_id}",
        "available_apis": ["get_user"],
        "max_steps": 3,
        "user_ids": ["U101", "U102", "U103", "U104", "U105"],
        "success_condition": lambda history, config: any(
            call["api_name"] == "get_user" and call.get("result", {}).get("success") for call in history
        ),
    },
    "medium": {
        "task_id": "TASK_MEDIUM",
        "description": "Generate invoice for user {user_id}",
        "available_apis": ["get_user", "get_orders", "get_product", "create_invoice"],
        "max_steps": 8,
        "user_ids": ["U101", "U102", "U103", "U104"],
        "success_condition": lambda history, config: any(
            call["api_name"] == "create_invoice" and call.get("result", {}).get("success")
            for call in history
        ),
    },
    "hard": {
        "task_id": "TASK_HARD",
        "description": "Resolve support ticket {ticket_id}: investigate, attempt refund if eligible, and send confirmation email",
        "available_apis": ["get_ticket", "get_user", "get_orders", "process_refund", "send_email"],
        "max_steps": 10,
        "ticket_ids": ["T301", "T302"],
        "success_condition": lambda history, config: _check_hard_success(history, config.get("ticket_id", "")),
    },
}


class ApiOpenEnvironment(Environment):
    """
    API Workflow Environment for multi-step task completion.

    The agent must learn to call APIs in the correct sequence to complete
    tasks like fetching user data, generating invoices, and processing refunds.

    Three difficulty levels:
    - Easy: Single API call (fetch user email)
    - Medium: Multi-step workflow (generate invoice)
    - Hard: Complex workflow with validation (process refund with constraints)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the API Workflow environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_config: Dict[str, Any] = {}
        self._api_call_history: List[Dict[str, Any]] = []
        self._task_complete = False
        self._last_api_result: Optional[Dict[str, Any]] = None

    def reset(self, task_difficulty: str = "random") -> ApiOpenObservation:
        """
        Reset the environment and select a new task.

        Args:
            task_difficulty: "easy", "medium", "hard", or "random"

        Returns:
            ApiOpenObservation with the initial state
        """
        # Reset internal state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._api_call_history = []
        self._task_complete = False
        self._last_api_result = None
        reset_mock_db()

        # Select task
        if task_difficulty == "random":
            task_difficulty = random.choice(["easy", "medium", "hard"])
        elif task_difficulty not in TASKS:
            task_difficulty = "easy"

        task_template = TASKS[task_difficulty]

        # Parameterize task description
        if task_difficulty == "hard":
            ticket_id = random.choice(task_template["ticket_ids"])
            task_description = task_template["description"].format(ticket_id=ticket_id)
            self._task_config = {
                "task_id": task_template["task_id"],
                "difficulty": task_difficulty,
                "ticket_id": ticket_id,
            }
        else:
            user_id = random.choice(task_template["user_ids"])
            task_description = task_template["description"].format(user_id=user_id)
            self._task_config = {
                "task_id": task_template["task_id"],
                "difficulty": task_difficulty,
                "user_id": user_id,
            }

        self._task_config.update(
            {
                "description": task_description,
                "available_apis": task_template["available_apis"],
                "max_steps": task_template["max_steps"],
                "success_condition": task_template["success_condition"],
            }
        )

        return ApiOpenObservation(
            task_description=task_description,
            available_apis=task_template["available_apis"],
            last_api_result=None,
            api_call_history=[],
            step_count=0,
            max_steps=task_template["max_steps"],
            task_complete=False,
            done=False,
            reward=0.0,
        )

    def step(self, action: ApiOpenAction) -> ApiOpenObservation:  # type: ignore[override]
        """
        Execute an API call action.

        Args:
            action: ApiOpenAction with api_name and args

        Returns:
            ApiOpenObservation with updated state and reward
        """
        self._state.step_count += 1

        # Check if API is available for this task
        if action.api_name not in self._task_config["available_apis"]:
            result = {"success": False, "error": f"API '{action.api_name}' not available for this task"}
            reward = -0.3  # Penalty for invalid API
        else:
            # Call the API
            result = call_api(action.api_name, action.args)
            reward = self._calculate_reward(action.api_name, result)

        # Update history
        self._last_api_result = result
        call_record = {
            "api_name": action.api_name,
            "args": action.args,
            "result": result,
            "step": self._state.step_count,
        }
        self._api_call_history.append(call_record)

        # Check if task is complete
        self._task_complete = self._task_config["success_condition"](self._api_call_history, self._task_config)

        # Determine if episode is done
        done = self._task_complete or self._state.step_count >= self._task_config["max_steps"]

        # Bonus reward for completing task
        if self._task_complete:
            reward += 1.0

        return ApiOpenObservation(
            task_description=self._task_config["description"],
            available_apis=self._task_config["available_apis"],
            last_api_result=result,
            api_call_history=self._api_call_history.copy(),
            step_count=self._state.step_count,
            max_steps=self._task_config["max_steps"],
            task_complete=self._task_complete,
            done=done,
            reward=reward,
        )

    def _calculate_reward(self, api_name: str, result: Dict[str, Any]) -> float:
        """
        Calculate reward for an API call.

        Reward structure:
        - Base reward for successful API call: +0.1
        - Sequence progress reward: +0.2 for calling the next expected API
        - Critical API bonus: +0.2 for completing key workflow steps
        - Penalty for failed calls: -0.1
        - Step penalty: -0.02 per step (encourages efficiency)

        Args:
            api_name: Name of the API called
            result: Result from the API call

        Returns:
            Reward value (dense reward signal for learning)
        """
        # Step penalty to encourage efficiency
        reward = -0.02

        if not result.get("success", False):
            return reward - 0.1  # Penalty for failed API call

        # Base reward for successful call
        reward += 0.1

        # Calculate sequence progress reward
        difficulty = self._task_config.get("difficulty", "easy")
        if difficulty == "hard":
            ticket_id = self._task_config.get("ticket_id", "T301")
            expected_key = "hard_refund" if ticket_id == "T301" else "hard_no_refund"
        else:
            expected_key = difficulty

        expected_seq = EXPECTED_SEQUENCES.get(expected_key, [])
        successful_calls = [c["api_name"] for c in self._api_call_history if c.get("result", {}).get("success")]
        current_step = len(successful_calls)  # includes current call

        # Bonus for calling the next expected API in sequence
        if current_step <= len(expected_seq):
            if expected_seq[current_step - 1] == api_name:
                reward += 0.2  # Sequence progress reward

        # Additional bonus for critical workflow APIs
        critical_apis = {
            "create_invoice": 0.2,  # Final step in medium task
            "process_refund": 0.2,  # Important step in hard task
            "send_email": 0.15,     # Final step in hard task
        }
        reward += critical_apis.get(api_name, 0.0)

        return reward

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    def grade(self) -> float:
        """
        Compute a deterministic score for the episode in the strict range (0, 1).

        Scoring:
        - Task completion: 0.5 points
        - Efficiency (fewer steps): up to 0.3 points
        - Correct API sequence: up to 0.2 points
        """
        score = 0.0

        # Task completion bonus
        if self._task_complete:
            score += 0.5

        # Efficiency score
        difficulty = self._task_config.get("difficulty", "easy")
        if difficulty == "hard":
            ticket_id = self._task_config.get("ticket_id", "T301")
            expected_key = "hard_refund" if ticket_id == "T301" else "hard_no_refund"
        else:
            expected_key = difficulty

        expected_seq = EXPECTED_SEQUENCES.get(expected_key, [])
        optimal_steps = len(expected_seq)
        max_steps = self._task_config.get("max_steps", 10)
        actual_steps = self._state.step_count

        if actual_steps <= optimal_steps:
            efficiency = 1.0
        elif actual_steps >= max_steps:
            efficiency = 0.0
        else:
            efficiency = 1.0 - (actual_steps - optimal_steps) / max(1, max_steps - optimal_steps)

        score += 0.3 * efficiency

        # Sequence correctness score
        actual_apis = [c["api_name"] for c in self._api_call_history if c.get("result", {}).get("success")]
        matches = 0
        for i, api in enumerate(actual_apis):
            if i < len(expected_seq) and api == expected_seq[i]:
                matches += 1
            else:
                break

        sequence_score = matches / len(expected_seq) if expected_seq else 0
        score += 0.2 * sequence_score

        return _to_open_unit_interval(score)
