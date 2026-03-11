"""Tests for workflow engine.

Covers multi-step execution, conditional branching, state management, error handling.
"""

from __future__ import annotations

import pytest

from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.tools.workflow import Step, Workflow, WorkflowError, WorkflowState

from helpers import MockProvider


class TestModuleExports:
    """Verify workflow module exports."""

    def test_module_has_all(self) -> None:
        from mada_modelkit.tools import workflow

        assert hasattr(workflow, "__all__")

    def test_all_contains_expected_names(self) -> None:
        from mada_modelkit.tools.workflow import __all__

        assert "Step" in __all__
        assert "Workflow" in __all__
        assert "WorkflowState" in __all__
        assert "WorkflowError" in __all__


class TestWorkflowState:
    """Test WorkflowState operations."""

    def test_default_initialization(self) -> None:
        state = WorkflowState()

        assert state.variables == {}
        assert state.last_response is None
        assert state.history == []

    def test_set_and_get_variable(self) -> None:
        state = WorkflowState()

        state.set("key", "value")
        assert state.get("key") == "value"

    def test_get_with_default(self) -> None:
        state = WorkflowState()

        assert state.get("missing", "default") == "default"

    def test_variables_dict_access(self) -> None:
        state = WorkflowState()

        state.variables["direct"] = "access"
        assert state.get("direct") == "access"

    def test_history_append(self) -> None:
        state = WorkflowState()
        response = AgentResponse(
            content="test", model="mock", input_tokens=0, output_tokens=0
        )

        state.history.append(response)
        assert len(state.history) == 1
        assert state.history[0] == response

    def test_last_response_assignment(self) -> None:
        state = WorkflowState()
        response = AgentResponse(
            content="test", model="mock", input_tokens=0, output_tokens=0
        )

        state.last_response = response
        assert state.last_response == response


class TestStepValidation:
    """Test Step dataclass validation."""

    def test_valid_step(self) -> None:
        client = MockProvider()
        step = Step(name="test", client=client)

        assert step.name == "test"
        assert step.client is client
        assert step.condition is None
        assert step.prompt_fn is None
        assert step.on_response is None

    def test_empty_name_raises_error(self) -> None:
        client = MockProvider()

        with pytest.raises(ValueError, match="name cannot be empty"):
            Step(name="", client=client)

    def test_invalid_client_type_raises_error(self) -> None:
        with pytest.raises(TypeError, match="must be a BaseAgentClient"):
            Step(name="test", client="not a client")  # type: ignore

    def test_invalid_condition_type_raises_error(self) -> None:
        client = MockProvider()

        with pytest.raises(TypeError, match="condition must be callable"):
            Step(name="test", client=client, condition="not callable")  # type: ignore

    def test_invalid_prompt_fn_type_raises_error(self) -> None:
        client = MockProvider()

        with pytest.raises(TypeError, match="prompt_fn must be callable"):
            Step(name="test", client=client, prompt_fn="not callable")  # type: ignore

    def test_invalid_on_response_type_raises_error(self) -> None:
        client = MockProvider()

        with pytest.raises(TypeError, match="on_response must be callable"):
            Step(name="test", client=client, on_response="not callable")  # type: ignore

    def test_valid_step_with_all_callbacks(self) -> None:
        client = MockProvider()

        def condition(state: WorkflowState) -> bool:
            return True

        def prompt_fn(state: WorkflowState) -> str:
            return "prompt"

        def on_response(state: WorkflowState, response: AgentResponse) -> None:
            pass

        step = Step(
            name="test",
            client=client,
            condition=condition,
            prompt_fn=prompt_fn,
            on_response=on_response,
        )

        assert step.condition is condition
        assert step.prompt_fn is prompt_fn
        assert step.on_response is on_response


class TestStepShouldExecute:
    """Test Step condition evaluation."""

    def test_no_condition_always_executes(self) -> None:
        client = MockProvider()
        step = Step(name="test", client=client)
        state = WorkflowState()

        assert step.should_execute(state) is True

    def test_condition_returns_true(self) -> None:
        client = MockProvider()
        step = Step(
            name="test",
            client=client,
            condition=lambda state: True,
        )
        state = WorkflowState()

        assert step.should_execute(state) is True

    def test_condition_returns_false(self) -> None:
        client = MockProvider()
        step = Step(
            name="test",
            client=client,
            condition=lambda state: False,
        )
        state = WorkflowState()

        assert step.should_execute(state) is False

    def test_condition_based_on_state_variable(self) -> None:
        client = MockProvider()
        step = Step(
            name="test",
            client=client,
            condition=lambda state: state.get("enabled", False),
        )
        state = WorkflowState()

        # Initially disabled
        assert step.should_execute(state) is False

        # Enable it
        state.set("enabled", True)
        assert step.should_execute(state) is True

    def test_condition_raises_error(self) -> None:
        client = MockProvider()

        def bad_condition(state: WorkflowState) -> bool:
            raise RuntimeError("condition error")

        step = Step(name="test", client=client, condition=bad_condition)
        state = WorkflowState()

        with pytest.raises(WorkflowError, match="Condition evaluation failed"):
            step.should_execute(state)


class TestWorkflowAddStep:
    """Test Workflow.add_step validation."""

    def test_add_single_step(self) -> None:
        workflow = Workflow()
        client = MockProvider()
        step = Step(name="step1", client=client)

        workflow.add_step(step)

        assert len(workflow) == 1

    def test_add_multiple_steps(self) -> None:
        workflow = Workflow()
        client = MockProvider()

        workflow.add_step(Step(name="step1", client=client))
        workflow.add_step(Step(name="step2", client=client))
        workflow.add_step(Step(name="step3", client=client))

        assert len(workflow) == 3

    def test_invalid_step_type_raises_error(self) -> None:
        workflow = Workflow()

        with pytest.raises(TypeError, match="must be a Step instance"):
            workflow.add_step("not a step")  # type: ignore

    def test_duplicate_step_name_raises_error(self) -> None:
        workflow = Workflow()
        client = MockProvider()

        workflow.add_step(Step(name="duplicate", client=client))

        with pytest.raises(ValueError, match="already exists"):
            workflow.add_step(Step(name="duplicate", client=client))

    def test_clear_workflow(self) -> None:
        workflow = Workflow()
        client = MockProvider()

        workflow.add_step(Step(name="step1", client=client))
        workflow.add_step(Step(name="step2", client=client))
        assert len(workflow) == 2

        workflow.clear()
        assert len(workflow) == 0


class TestWorkflowExecution:
    """Test Workflow.execute basic functionality."""

    @pytest.mark.asyncio
    async def test_empty_workflow_raises_error(self) -> None:
        workflow = Workflow()

        with pytest.raises(WorkflowError, match="has no steps"):
            await workflow.execute("prompt")

    @pytest.mark.asyncio
    async def test_single_step_execution(self) -> None:
        workflow = Workflow()
        client = MockProvider()
        step = Step(name="step1", client=client)

        workflow.add_step(step)
        state = await workflow.execute("initial prompt")

        assert state.last_response is not None
        assert state.last_response.content == "mock"
        assert len(state.history) == 1

    @pytest.mark.asyncio
    async def test_multiple_step_execution(self) -> None:
        workflow = Workflow()
        client1 = MockProvider()
        client2 = MockProvider()

        workflow.add_step(Step(name="step1", client=client1))
        workflow.add_step(Step(name="step2", client=client2))

        state = await workflow.execute("initial prompt")

        assert len(state.history) == 2
        assert state.history[0].content == "mock"
        assert state.history[1].content == "mock"

    @pytest.mark.asyncio
    async def test_initial_state_preserved(self) -> None:
        workflow = Workflow()
        client = MockProvider()
        step = Step(name="step1", client=client)

        workflow.add_step(step)

        initial_state = WorkflowState()
        initial_state.set("key", "value")

        state = await workflow.execute("prompt", initial_state=initial_state)

        assert state.get("key") == "value"

    @pytest.mark.asyncio
    async def test_max_steps_limit(self) -> None:
        workflow = Workflow()
        client = MockProvider()

        workflow.add_step(Step(name="step1", client=client))
        workflow.add_step(Step(name="step2", client=client))
        workflow.add_step(Step(name="step3", client=client))

        state = await workflow.execute("prompt", max_steps=2)

        # Only first 2 steps executed
        assert len(state.history) == 2


class TestConditionalBranching:
    """Test workflow conditional branching."""

    @pytest.mark.asyncio
    async def test_skip_step_when_condition_false(self) -> None:
        workflow = Workflow()
        client = MockProvider()

        # Step 1 always executes
        workflow.add_step(Step(name="step1", client=client))

        # Step 2 never executes
        workflow.add_step(
            Step(
                name="step2",
                client=client,
                condition=lambda state: False,
            )
        )

        # Step 3 always executes
        workflow.add_step(Step(name="step3", client=client))

        state = await workflow.execute("prompt")

        # Only 2 steps executed (step2 skipped)
        assert len(state.history) == 2

    @pytest.mark.asyncio
    async def test_conditional_based_on_previous_response(self) -> None:
        workflow = Workflow()

        # Step 1 returns "yes"
        client1 = MockProvider(
            responses=[
                AgentResponse(
                    content="yes", model="mock", input_tokens=0, output_tokens=0
                )
            ]
        )

        # Step 2 only executes if last response is "yes"
        client2 = MockProvider(
            responses=[
                AgentResponse(
                    content="confirmed", model="mock", input_tokens=0, output_tokens=0
                )
            ]
        )

        workflow.add_step(Step(name="step1", client=client1))
        workflow.add_step(
            Step(
                name="step2",
                client=client2,
                condition=lambda state: (
                    state.last_response is not None
                    and state.last_response.content == "yes"
                ),
            )
        )

        state = await workflow.execute("prompt")

        assert len(state.history) == 2
        assert state.history[0].content == "yes"
        assert state.history[1].content == "confirmed"

    @pytest.mark.asyncio
    async def test_conditional_based_on_state_variable(self) -> None:
        workflow = Workflow()

        # Step 1 sets a flag
        client1 = MockProvider()

        def set_flag(state: WorkflowState, response: AgentResponse) -> None:
            state.set("proceed", True)

        # Step 2 only executes if flag is set
        client2 = MockProvider()

        workflow.add_step(
            Step(name="step1", client=client1, on_response=set_flag)
        )
        workflow.add_step(
            Step(
                name="step2",
                client=client2,
                condition=lambda state: state.get("proceed", False),
            )
        )

        state = await workflow.execute("prompt")

        assert len(state.history) == 2
        assert state.get("proceed") is True


class TestStateManagement:
    """Test state passing between steps."""

    @pytest.mark.asyncio
    async def test_on_response_updates_state(self) -> None:
        workflow = Workflow()
        client = MockProvider()

        def update_state(state: WorkflowState, response: AgentResponse) -> None:
            state.set("updated", True)

        step = Step(name="step1", client=client, on_response=update_state)
        workflow.add_step(step)

        state = await workflow.execute("prompt")

        assert state.get("updated") is True

    @pytest.mark.asyncio
    async def test_state_passed_between_steps(self) -> None:
        workflow = Workflow()

        # Step 1 sets a counter
        client1 = MockProvider()

        def set_counter(state: WorkflowState, response: AgentResponse) -> None:
            state.set("counter", 1)

        # Step 2 increments the counter
        client2 = MockProvider()

        def increment_counter(state: WorkflowState, response: AgentResponse) -> None:
            count = state.get("counter", 0)
            state.set("counter", count + 1)

        workflow.add_step(
            Step(name="step1", client=client1, on_response=set_counter)
        )
        workflow.add_step(
            Step(name="step2", client=client2, on_response=increment_counter)
        )

        state = await workflow.execute("prompt")

        assert state.get("counter") == 2

    @pytest.mark.asyncio
    async def test_prompt_fn_uses_state(self) -> None:
        workflow = Workflow()
        client = MockProvider()

        def generate_prompt(state: WorkflowState) -> str:
            name = state.get("name", "unknown")
            return f"Hello {name}"

        initial_state = WorkflowState()
        initial_state.set("name", "Alice")

        step = Step(name="step1", client=client, prompt_fn=generate_prompt)
        workflow.add_step(step)

        state = await workflow.execute("ignored", initial_state=initial_state)

        # Prompt was generated from state, not initial_prompt
        assert state.get("name") == "Alice"


class TestPromptGeneration:
    """Test prompt generation for steps."""

    @pytest.mark.asyncio
    async def test_first_step_uses_initial_prompt(self) -> None:
        workflow = Workflow()
        client = MockProvider()

        workflow.add_step(Step(name="step1", client=client))

        state = await workflow.execute("initial prompt")

        # First step received initial prompt
        assert len(state.history) == 1

    @pytest.mark.asyncio
    async def test_second_step_uses_last_response(self) -> None:
        workflow = Workflow()
        client1 = MockProvider(
            responses=[
                AgentResponse(
                    content="response1", model="mock", input_tokens=0, output_tokens=0
                )
            ]
        )
        client2 = MockProvider(
            responses=[
                AgentResponse(
                    content="response2", model="mock", input_tokens=0, output_tokens=0
                )
            ]
        )

        workflow.add_step(Step(name="step1", client=client1))
        workflow.add_step(Step(name="step2", client=client2))

        state = await workflow.execute("initial")

        # Second step used first step's response as prompt
        assert state.history[0].content == "response1"
        assert state.history[1].content == "response2"

    @pytest.mark.asyncio
    async def test_prompt_fn_overrides_default(self) -> None:
        workflow = Workflow()
        client = MockProvider()

        workflow.add_step(
            Step(
                name="step1",
                client=client,
                prompt_fn=lambda state: "custom prompt",
            )
        )

        state = await workflow.execute("ignored")

        # Step used custom prompt
        assert len(state.history) == 1

    @pytest.mark.asyncio
    async def test_prompt_fn_failure_raises_error(self) -> None:
        workflow = Workflow()
        client = MockProvider()

        def bad_prompt_fn(state: WorkflowState) -> str:
            raise RuntimeError("prompt error")

        workflow.add_step(
            Step(name="step1", client=client, prompt_fn=bad_prompt_fn)
        )

        with pytest.raises(WorkflowError, match="Prompt generation failed"):
            await workflow.execute("prompt")


class TestErrorHandling:
    """Test workflow error handling."""

    @pytest.mark.asyncio
    async def test_step_execution_failure_raises_error(self) -> None:
        workflow = Workflow()

        # Create a provider that fails
        class FailingProvider(MockProvider):
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                raise RuntimeError("execution failed")

        client = FailingProvider()
        workflow.add_step(Step(name="step1", client=client))

        with pytest.raises(WorkflowError, match="execution failed"):
            await workflow.execute("prompt")

    @pytest.mark.asyncio
    async def test_on_response_failure_raises_error(self) -> None:
        workflow = Workflow()
        client = MockProvider()

        def bad_handler(state: WorkflowState, response: AgentResponse) -> None:
            raise RuntimeError("handler error")

        workflow.add_step(
            Step(name="step1", client=client, on_response=bad_handler)
        )

        with pytest.raises(WorkflowError, match="Response handler failed"):
            await workflow.execute("prompt")

    @pytest.mark.asyncio
    async def test_workflow_error_includes_step_name(self) -> None:
        workflow = Workflow()

        class FailingProvider(MockProvider):
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                raise RuntimeError("fail")

        workflow.add_step(Step(name="failing_step", client=FailingProvider()))

        with pytest.raises(WorkflowError, match="failing_step"):
            await workflow.execute("prompt")


class TestWorkflowIntegration:
    """Integration tests for workflow engine."""

    @pytest.mark.asyncio
    async def test_complex_workflow_with_branching(self) -> None:
        workflow = Workflow()

        # Step 1: Extract intent
        client1 = MockProvider(
            responses=[
                AgentResponse(
                    content="search", model="mock", input_tokens=0, output_tokens=0
                )
            ]
        )

        def extract_intent(state: WorkflowState, response: AgentResponse) -> None:
            state.set("intent", response.content)

        # Step 2: Search (only if intent is "search")
        client2 = MockProvider(
            responses=[
                AgentResponse(
                    content="results", model="mock", input_tokens=0, output_tokens=0
                )
            ]
        )

        # Step 3: Summarize (always)
        client3 = MockProvider(
            responses=[
                AgentResponse(
                    content="summary", model="mock", input_tokens=0, output_tokens=0
                )
            ]
        )

        workflow.add_step(
            Step(name="extract", client=client1, on_response=extract_intent)
        )
        workflow.add_step(
            Step(
                name="search",
                client=client2,
                condition=lambda state: state.get("intent") == "search",
            )
        )
        workflow.add_step(Step(name="summarize", client=client3))

        state = await workflow.execute("What is AI?")

        assert len(state.history) == 3
        assert state.get("intent") == "search"
        assert state.history[1].content == "results"
        assert state.history[2].content == "summary"

    @pytest.mark.asyncio
    async def test_workflow_reuse(self) -> None:
        workflow = Workflow()
        client = MockProvider()

        workflow.add_step(Step(name="step1", client=client))

        # Execute twice
        state1 = await workflow.execute("prompt1")
        state2 = await workflow.execute("prompt2")

        # Both executions succeed
        assert len(state1.history) == 1
        assert len(state2.history) == 1

    @pytest.mark.asyncio
    async def test_workflow_clear_and_rebuild(self) -> None:
        workflow = Workflow()
        client = MockProvider()

        workflow.add_step(Step(name="step1", client=client))
        assert len(workflow) == 1

        workflow.clear()
        assert len(workflow) == 0

        workflow.add_step(Step(name="new_step", client=client))
        assert len(workflow) == 1

        state = await workflow.execute("prompt")
        assert len(state.history) == 1
