# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the API Workflow Environment.

This environment simulates a real-world API workflow system where an agent
must call APIs in the correct sequence to complete tasks.
"""

import json
from typing import Dict, List, Any, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator


class ApiOpenAction(Action):
    """
    Action for the API Workflow Environment.

    The agent specifies which API to call and what arguments to pass.
    """

    api_name: str = Field(..., description="Name of the API to call")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the API")

    @field_validator("args", mode="before")
    @classmethod
    def coerce_args_to_dict(cls, value: Any) -> Dict[str, Any]:
        """Accept both dict args and JSON-string args from the web UI."""
        if value is None:
            return {}

        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("args must be a JSON object string") from exc
            if not isinstance(parsed, dict):
                raise ValueError("args JSON must decode to an object")
            return parsed

        raise ValueError("args must be a dictionary or JSON object string")


class ApiOpenObservation(Observation):
    """
    Observation from the API Workflow Environment.

    Contains the task description, available APIs, last API result,
    call history, and episode state.
    """

    task_description: str = Field(default="", description="Description of the task to complete")
    available_apis: List[str] = Field(default_factory=list, description="List of available API names")
    last_api_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Result from the last API call"
    )
    api_call_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of all API calls made"
    )
    step_count: int = Field(default=0, description="Current step number in the episode")
    max_steps: int = Field(default=10, description="Maximum steps allowed")
    task_complete: bool = Field(default=False, description="Whether the task has been completed")
