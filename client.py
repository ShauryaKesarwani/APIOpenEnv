"""Api Open Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import ApiOpenAction, ApiOpenObservation


class ApiOpenEnv(
    EnvClient[ApiOpenAction, ApiOpenObservation, State]
):
    """
    Client for the API Workflow Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with ApiOpenEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_description)
        ...
        ...     result = client.step(ApiOpenAction(api_name="get_user", args={"user_id": "U101"}))
        ...     print(result.observation.last_api_result)
    """

    def _step_payload(self, action: ApiOpenAction) -> Dict:
        """Convert ApiOpenAction to JSON payload for step message."""
        return {
            "api_name": action.api_name,
            "args": action.args,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ApiOpenObservation]:
        """Parse server response into StepResult[ApiOpenObservation]."""
        obs_data = payload.get("observation", {})
        observation = ApiOpenObservation(
            task_description=obs_data.get("task_description", ""),
            available_apis=obs_data.get("available_apis", []),
            last_api_result=obs_data.get("last_api_result"),
            api_call_history=obs_data.get("api_call_history", []),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 10),
            task_complete=obs_data.get("task_complete", False),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
