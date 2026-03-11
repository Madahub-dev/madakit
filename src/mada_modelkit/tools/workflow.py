"""Workflow engine for multi-step agent workflows.

Conditional branching, state management, and error handling.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._errors import MiddlewareError
from mada_modelkit._types import AgentRequest, AgentResponse

__all__ = ["Step", "Workflow", "WorkflowState", "WorkflowError"]


class WorkflowError(MiddlewareError):
    """Raised when workflow execution fails."""

    pass


@dataclass
class WorkflowState:
    """State passed between workflow steps.

    Attributes:
        variables: Dictionary of variables available to all steps.
        last_response: Last agent response from previous step.
        history: List of all responses from previous steps.
    """

    variables: dict[str, Any] = field(default_factory=dict)
    last_response: Optional[AgentResponse] = None
    history: list[AgentResponse] = field(default_factory=list)

    def set(self, key: str, value: Any) -> None:
        """Set a variable in the workflow state.

        Args:
            key: Variable name.
            value: Variable value.
        """
        self.variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable from the workflow state.

        Args:
            key: Variable name.
            default: Default value if key not found.

        Returns:
            Variable value or default.
        """
        return self.variables.get(key, default)


@dataclass
class Step:
    """A single step in a workflow.

    Attributes:
        name: Step name for identification.
        client: Agent client to execute this step.
        condition: Optional condition that must be True to execute this step.
        prompt_fn: Function that generates prompt from workflow state.
        on_response: Optional callback to process response and update state.
    """

    name: str
    client: BaseAgentClient
    condition: Optional[Callable[[WorkflowState], bool]] = None
    prompt_fn: Optional[Callable[[WorkflowState], str]] = None
    on_response: Optional[Callable[[WorkflowState, AgentResponse], None]] = None

    def __post_init__(self) -> None:
        """Validate step configuration."""
        if not self.name:
            raise ValueError("Step name cannot be empty")

        if not isinstance(self.client, BaseAgentClient):
            raise TypeError("client must be a BaseAgentClient instance")

        if self.condition is not None and not callable(self.condition):
            raise TypeError("condition must be callable")

        if self.prompt_fn is not None and not callable(self.prompt_fn):
            raise TypeError("prompt_fn must be callable")

        if self.on_response is not None and not callable(self.on_response):
            raise TypeError("on_response must be callable")

    def should_execute(self, state: WorkflowState) -> bool:
        """Check if this step should execute based on condition.

        Args:
            state: Current workflow state.

        Returns:
            True if step should execute, False otherwise.
        """
        if self.condition is None:
            return True

        try:
            return bool(self.condition(state))
        except Exception as e:
            raise WorkflowError(
                f"Condition evaluation failed for step '{self.name}': {e}"
            ) from e


class Workflow:
    """Multi-step agent workflow with conditional branching.

    Executes a series of steps, passing state between them.
    """

    def __init__(self) -> None:
        """Initialize empty workflow."""
        self._steps: list[Step] = []

    def add_step(self, step: Step) -> None:
        """Add a step to the workflow.

        Args:
            step: Step to add.

        Raises:
            TypeError: If step is not a Step instance.
            ValueError: If step name already exists.
        """
        if not isinstance(step, Step):
            raise TypeError("step must be a Step instance")

        # Check for duplicate names
        if any(s.name == step.name for s in self._steps):
            raise ValueError(f"Step with name '{step.name}' already exists")

        self._steps.append(step)

    async def execute(
        self,
        initial_prompt: str,
        initial_state: Optional[WorkflowState] = None,
        max_steps: Optional[int] = None,
    ) -> WorkflowState:
        """Execute the workflow.

        Args:
            initial_prompt: Initial prompt for the first step.
            initial_state: Optional initial state with variables.
            max_steps: Maximum number of steps to execute (default: all steps).

        Returns:
            Final workflow state after execution.

        Raises:
            WorkflowError: If workflow execution fails.
        """
        if not self._steps:
            raise WorkflowError("Workflow has no steps")

        # Initialize state
        state = initial_state if initial_state is not None else WorkflowState()

        # Track executed steps
        executed_count = 0
        step_limit = max_steps if max_steps is not None else len(self._steps)

        for step in self._steps:
            # Check max steps limit
            if executed_count >= step_limit:
                break

            # Check if step should execute
            if not step.should_execute(state):
                continue

            # Generate prompt for this step
            if step.prompt_fn is not None:
                try:
                    prompt = step.prompt_fn(state)
                except Exception as e:
                    raise WorkflowError(
                        f"Prompt generation failed for step '{step.name}': {e}"
                    ) from e
            else:
                # Use initial prompt for first executed step, last response for others
                if executed_count == 0:
                    prompt = initial_prompt
                elif state.last_response is not None:
                    prompt = state.last_response.content
                else:
                    raise WorkflowError(
                        f"No prompt available for step '{step.name}'"
                    )

            # Execute step
            try:
                request = AgentRequest(prompt=prompt)
                response = await step.client.send_request(request)
            except Exception as e:
                raise WorkflowError(
                    f"Step '{step.name}' execution failed: {e}"
                ) from e

            # Update state
            state.last_response = response
            state.history.append(response)

            # Call response handler if provided
            if step.on_response is not None:
                try:
                    step.on_response(state, response)
                except Exception as e:
                    raise WorkflowError(
                        f"Response handler failed for step '{step.name}': {e}"
                    ) from e

            executed_count += 1

        return state

    def clear(self) -> None:
        """Remove all steps from the workflow."""
        self._steps.clear()

    def __len__(self) -> int:
        """Return number of steps in the workflow."""
        return len(self._steps)
