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

from typing import Dict, List, Any, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ApiOpenAction(Action):
    """
    Action for the API Workflow Environment.

    The agent specifies which API to call and what arguments to pass.
    """

    api_name: str = Field(..., description="Name of the API to call")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the API")


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
