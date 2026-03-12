"""Tests for function calling middleware.

Tests FunctionCallingMiddleware tool detection, execution, result injection,
and multi-step iteration.
"""

from __future__ import annotations

import pytest

from madakit._errors import MiddlewareError
from madakit._types import AgentRequest, AgentResponse, StreamChunk
from madakit.middleware.function_calling import FunctionCallingMiddleware
from madakit.tools.registry import ToolRegistry

from helpers import MockProvider


class TestModuleExports:
    """Test module-level exports and imports."""

    def test_all_exports(self) -> None:
        """__all__ contains only FunctionCallingMiddleware."""
        from madakit.middleware import function_calling

        assert function_calling.__all__ == ["FunctionCallingMiddleware"]

    def test_middleware_importable(self) -> None:
        """FunctionCallingMiddleware can be imported from module."""
        from madakit.middleware.function_calling import (
            FunctionCallingMiddleware as FCM,
        )

        assert FCM is not None


class TestFunctionCallingMiddlewareConstructor:
    """Test FunctionCallingMiddleware constructor and initialization."""

    def test_minimal_constructor(self) -> None:
        """Constructor accepts client and registry with default max_iterations."""
        mock = MockProvider()
        registry = ToolRegistry()

        middleware = FunctionCallingMiddleware(client=mock, registry=registry)

        assert middleware._client is mock
        assert middleware._registry is registry
        assert middleware._max_iterations == 3

    def test_with_custom_max_iterations(self) -> None:
        """Constructor accepts custom max_iterations."""
        mock = MockProvider()
        registry = ToolRegistry()

        middleware = FunctionCallingMiddleware(
            client=mock,
            registry=registry,
            max_iterations=5,
        )

        assert middleware._max_iterations == 5

    def test_zero_max_iterations_raises(self) -> None:
        """Zero max_iterations raises ValueError."""
        mock = MockProvider()
        registry = ToolRegistry()

        with pytest.raises(ValueError, match="max_iterations must be positive"):
            FunctionCallingMiddleware(
                client=mock,
                registry=registry,
                max_iterations=0,
            )

    def test_negative_max_iterations_raises(self) -> None:
        """Negative max_iterations raises ValueError."""
        mock = MockProvider()
        registry = ToolRegistry()

        with pytest.raises(ValueError, match="max_iterations must be positive"):
            FunctionCallingMiddleware(
                client=mock,
                registry=registry,
                max_iterations=-1,
            )

    def test_super_init_called(self) -> None:
        """FunctionCallingMiddleware calls BaseAgentClient.__init__."""
        mock = MockProvider()
        registry = ToolRegistry()
        middleware = FunctionCallingMiddleware(client=mock, registry=registry)

        assert hasattr(middleware, "send_request")
        assert hasattr(middleware, "send_request_stream")


class TestToolDetection:
    """Test tool call detection in responses."""

    def test_detect_single_tool_call(self) -> None:
        """Single tool call is detected."""
        middleware = FunctionCallingMiddleware(
            client=MockProvider(),
            registry=ToolRegistry(),
        )

        content = 'I will call a tool: <tool_call name="get_weather">{"city": "NYC"}</tool_call>'

        tool_calls = middleware._detect_tool_calls(content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["arguments"] == {"city": "NYC"}

    def test_detect_multiple_tool_calls(self) -> None:
        """Multiple tool calls are detected."""
        middleware = FunctionCallingMiddleware(
            client=MockProvider(),
            registry=ToolRegistry(),
        )

        content = '''
        <tool_call name="tool1">{"arg1": "value1"}</tool_call>
        Some text in between.
        <tool_call name="tool2">{"arg2": "value2"}</tool_call>
        '''

        tool_calls = middleware._detect_tool_calls(content)

        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "tool1"
        assert tool_calls[1]["name"] == "tool2"

    def test_detect_no_tool_calls(self) -> None:
        """Returns empty list when no tool calls present."""
        middleware = FunctionCallingMiddleware(
            client=MockProvider(),
            registry=ToolRegistry(),
        )

        content = "This is a normal response with no tool calls."

        tool_calls = middleware._detect_tool_calls(content)

        assert tool_calls == []

    def test_detect_empty_arguments(self) -> None:
        """Tool call with empty arguments is detected."""
        middleware = FunctionCallingMiddleware(
            client=MockProvider(),
            registry=ToolRegistry(),
        )

        content = '<tool_call name="no_args"></tool_call>'

        tool_calls = middleware._detect_tool_calls(content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "no_args"
        assert tool_calls[0]["arguments"] == {}

    def test_detect_invalid_json_skipped(self) -> None:
        """Tool call with invalid JSON is skipped."""
        middleware = FunctionCallingMiddleware(
            client=MockProvider(),
            registry=ToolRegistry(),
        )

        content = '<tool_call name="bad">invalid json</tool_call>'

        tool_calls = middleware._detect_tool_calls(content)

        assert tool_calls == []

    def test_detect_multiline_arguments(self) -> None:
        """Tool call with multiline JSON is detected."""
        middleware = FunctionCallingMiddleware(
            client=MockProvider(),
            registry=ToolRegistry(),
        )

        content = '''<tool_call name="complex">{
            "arg1": "value1",
            "arg2": "value2"
        }</tool_call>'''

        tool_calls = middleware._detect_tool_calls(content)

        assert len(tool_calls) == 1
        assert tool_calls[0]["arguments"] == {"arg1": "value1", "arg2": "value2"}


class TestToolExecution:
    """Test tool execution."""

    def test_execute_simple_tool(self) -> None:
        """Simple tool execution works."""
        registry = ToolRegistry()

        def greet(name: str) -> str:
            return f"Hello {name}"

        registry.register_function(
            name="greet",
            function=greet,
            description="Greet someone",
        )

        middleware = FunctionCallingMiddleware(
            client=MockProvider(),
            registry=registry,
        )

        result = middleware._execute_tool("greet", {"name": "Alice"})

        assert result == "Hello Alice"

    def test_execute_tool_with_validation(self) -> None:
        """Tool execution validates arguments."""
        registry = ToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        registry.register_function(
            name="add",
            function=add,
            description="Add numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        )

        middleware = FunctionCallingMiddleware(
            client=MockProvider(),
            registry=registry,
        )

        # Valid execution
        result = middleware._execute_tool("add", {"a": 5, "b": 3})
        assert result == 8

        # Missing required argument
        with pytest.raises(MiddlewareError, match="Missing required parameter"):
            middleware._execute_tool("add", {"a": 5})

    def test_execute_nonexistent_tool_raises(self) -> None:
        """Executing nonexistent tool raises MiddlewareError."""
        registry = ToolRegistry()
        middleware = FunctionCallingMiddleware(
            client=MockProvider(),
            registry=registry,
        )

        with pytest.raises(MiddlewareError, match="Tool 'missing' not found"):
            middleware._execute_tool("missing", {})

    def test_execute_tool_exception_wrapped(self) -> None:
        """Tool execution exception is wrapped in MiddlewareError."""
        registry = ToolRegistry()

        def failing_tool():
            raise ValueError("Tool failed")

        registry.register_function(
            name="fail",
            function=failing_tool,
            description="Failing tool",
        )

        middleware = FunctionCallingMiddleware(
            client=MockProvider(),
            registry=registry,
        )

        with pytest.raises(MiddlewareError, match="execution failed"):
            middleware._execute_tool("fail", {})


class TestResultInjection:
    """Test tool result injection."""

    def test_inject_single_result(self) -> None:
        """Single tool result is injected into prompt."""
        middleware = FunctionCallingMiddleware(
            client=MockProvider(),
            registry=ToolRegistry(),
        )

        new_prompt = middleware._inject_tool_results(
            original_prompt="What's the weather?",
            response_content='<tool_call name="get_weather">{"city": "NYC"}</tool_call>',
            tool_results=[{"name": "get_weather", "result": "Sunny, 72F"}],
        )

        assert "Previous conversation:" in new_prompt
        assert "User: What's the weather?" in new_prompt
        assert "Assistant: <tool_call" in new_prompt
        assert "Tool results:" in new_prompt
        assert "get_weather" in new_prompt
        assert "Sunny, 72F" in new_prompt
        assert "Please continue" in new_prompt

    def test_inject_multiple_results(self) -> None:
        """Multiple tool results are injected."""
        middleware = FunctionCallingMiddleware(
            client=MockProvider(),
            registry=ToolRegistry(),
        )

        new_prompt = middleware._inject_tool_results(
            original_prompt="Get info",
            response_content="Calling tools",
            tool_results=[
                {"name": "tool1", "result": "result1"},
                {"name": "tool2", "result": "result2"},
            ],
        )

        assert "tool1" in new_prompt
        assert "result1" in new_prompt
        assert "tool2" in new_prompt
        assert "result2" in new_prompt


class TestAutoFunctionCalling:
    """Test automatic function calling in send_request."""

    @pytest.mark.asyncio
    async def test_no_tool_calls_passthrough(self) -> None:
        """Response without tool calls is returned as-is."""
        mock = MockProvider(
            responses=[
                AgentResponse(
                    content="Normal response",
                    model="test",
                    input_tokens=10,
                    output_tokens=5,
                )
            ]
        )
        registry = ToolRegistry()
        middleware = FunctionCallingMiddleware(client=mock, registry=registry)

        response = await middleware.send_request(AgentRequest(prompt="test"))

        assert response.content == "Normal response"

    @pytest.mark.asyncio
    async def test_single_iteration_tool_call(self) -> None:
        """Single tool call is executed and result returned."""
        registry = ToolRegistry()

        def get_time() -> str:
            return "12:00 PM"

        registry.register_function(
            name="get_time",
            function=get_time,
            description="Get current time",
        )

        # First response has tool call, second has final answer
        mock = MockProvider(
            responses=[
                AgentResponse(
                    content='<tool_call name="get_time">{}</tool_call>',
                    model="test",
                    input_tokens=10,
                    output_tokens=5,
                ),
                AgentResponse(
                    content="The time is 12:00 PM",
                    model="test",
                    input_tokens=15,
                    output_tokens=8,
                ),
            ]
        )

        middleware = FunctionCallingMiddleware(client=mock, registry=registry)

        response = await middleware.send_request(AgentRequest(prompt="What time is it?"))

        assert response.content == "The time is 12:00 PM"
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_iterations(self) -> None:
        """Multiple iterations of tool calls work."""
        registry = ToolRegistry()

        def step1() -> str:
            return "step1_result"

        def step2() -> str:
            return "step2_result"

        registry.register_function(
            name="step1",
            function=step1,
            description="First step",
        )
        registry.register_function(
            name="step2",
            function=step2,
            description="Second step",
        )

        mock = MockProvider(
            responses=[
                AgentResponse(
                    content='<tool_call name="step1">{}</tool_call>',
                    model="test",
                    input_tokens=10,
                    output_tokens=5,
                ),
                AgentResponse(
                    content='<tool_call name="step2">{}</tool_call>',
                    model="test",
                    input_tokens=15,
                    output_tokens=8,
                ),
                AgentResponse(
                    content="Done",
                    model="test",
                    input_tokens=20,
                    output_tokens=2,
                ),
            ]
        )

        middleware = FunctionCallingMiddleware(client=mock, registry=registry)

        response = await middleware.send_request(AgentRequest(prompt="Run workflow"))

        assert response.content == "Done"
        assert mock.call_count == 3

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self) -> None:
        """Raises MiddlewareError when max iterations exceeded."""
        registry = ToolRegistry()

        def infinite_tool() -> str:
            return "continue"

        registry.register_function(
            name="infinite",
            function=infinite_tool,
            description="Infinite tool",
        )

        # Always return tool call
        class InfiniteProvider(MockProvider):
            async def send_request(self, request):
                return AgentResponse(
                    content='<tool_call name="infinite">{}</tool_call>',
                    model="test",
                    input_tokens=10,
                    output_tokens=5,
                )

        middleware = FunctionCallingMiddleware(
            client=InfiniteProvider(),
            registry=registry,
            max_iterations=2,
        )

        with pytest.raises(MiddlewareError, match="Max iterations.*reached"):
            await middleware.send_request(AgentRequest(prompt="test"))

    @pytest.mark.asyncio
    async def test_multiple_tools_in_one_response(self) -> None:
        """Multiple tool calls in one response are all executed."""
        registry = ToolRegistry()

        def tool_a() -> str:
            return "a_result"

        def tool_b() -> str:
            return "b_result"

        registry.register_function(name="tool_a", function=tool_a, description="Tool A")
        registry.register_function(name="tool_b", function=tool_b, description="Tool B")

        mock = MockProvider(
            responses=[
                AgentResponse(
                    content='<tool_call name="tool_a">{}</tool_call> and <tool_call name="tool_b">{}</tool_call>',
                    model="test",
                    input_tokens=10,
                    output_tokens=5,
                ),
                AgentResponse(
                    content="Both tools completed",
                    model="test",
                    input_tokens=15,
                    output_tokens=8,
                ),
            ]
        )

        middleware = FunctionCallingMiddleware(client=mock, registry=registry)

        response = await middleware.send_request(AgentRequest(prompt="test"))

        assert response.content == "Both tools completed"


class TestStreaming:
    """Test streaming behavior with function calling."""

    @pytest.mark.asyncio
    async def test_send_request_stream_converts_to_non_streaming(self) -> None:
        """send_request_stream converts to non-streaming internally."""
        registry = ToolRegistry()

        def get_data() -> str:
            return "data"

        registry.register_function(
            name="get_data",
            function=get_data,
            description="Get data",
        )

        mock = MockProvider(
            responses=[
                AgentResponse(
                    content='<tool_call name="get_data">{}</tool_call>',
                    model="test",
                    input_tokens=10,
                    output_tokens=5,
                ),
                AgentResponse(
                    content="Final response",
                    model="test",
                    input_tokens=15,
                    output_tokens=8,
                ),
            ]
        )

        middleware = FunctionCallingMiddleware(client=mock, registry=registry)

        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="test")):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert chunks[0].delta == "Final response"


class TestIntegration:
    """Test full integration scenarios."""

    @pytest.mark.asyncio
    async def test_calculator_workflow(self) -> None:
        """Full calculator workflow with tool calling."""
        registry = ToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        def multiply(a: int, b: int) -> int:
            return a * b

        registry.register_function(
            name="add",
            function=add,
            description="Add numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
            },
        )
        registry.register_function(
            name="multiply",
            function=multiply,
            description="Multiply numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
            },
        )

        mock = MockProvider(
            responses=[
                AgentResponse(
                    content='<tool_call name="add">{"a": 5, "b": 3}</tool_call>',
                    model="test",
                    input_tokens=10,
                    output_tokens=5,
                ),
                AgentResponse(
                    content='<tool_call name="multiply">{"a": 8, "b": 2}</tool_call>',
                    model="test",
                    input_tokens=15,
                    output_tokens=8,
                ),
                AgentResponse(
                    content="The result is 16",
                    model="test",
                    input_tokens=20,
                    output_tokens=5,
                ),
            ]
        )

        middleware = FunctionCallingMiddleware(client=mock, registry=registry)

        response = await middleware.send_request(
            AgentRequest(prompt="Add 5 and 3, then multiply the result by 2")
        )

        assert response.content == "The result is 16"

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """FunctionCallingMiddleware works as context manager."""
        registry = ToolRegistry()
        mock = MockProvider()
        middleware = FunctionCallingMiddleware(client=mock, registry=registry)

        async with middleware:
            response = await middleware.send_request(AgentRequest(prompt="test"))
            assert response.content == "mock"
