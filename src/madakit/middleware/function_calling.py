"""Function calling middleware for madakit.

Automatic tool detection, execution, and result injection.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

from madakit._base import BaseAgentClient
from madakit._errors import MiddlewareError
from madakit._types import AgentRequest, AgentResponse, StreamChunk
from madakit.tools.registry import ToolRegistry

__all__ = ["FunctionCallingMiddleware"]


class FunctionCallingMiddleware(BaseAgentClient):
    """Middleware for automatic function calling.

    Detects tool calls in responses, executes them, and injects results back.
    """

    # Pattern for detecting tool calls in response content
    # Supports formats like: <tool_call name="function_name">{"arg": "value"}</tool_call>
    _TOOL_CALL_PATTERN = re.compile(
        r'<tool_call\s+name="([^"]+)">(.*?)</tool_call>',
        re.DOTALL,
    )

    def __init__(
        self,
        client: BaseAgentClient,
        registry: ToolRegistry,
        max_iterations: int = 3,
    ) -> None:
        """Initialise with client, tool registry, and iteration limit.

        Args:
            client: Wrapped client to delegate to.
            registry: Tool registry for looking up and executing tools.
            max_iterations: Maximum number of tool call iterations (default 3).

        Raises:
            ValueError: If max_iterations is not positive.
        """
        super().__init__()
        self._client = client
        self._registry = registry

        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        self._max_iterations = max_iterations

    def _detect_tool_calls(self, content: str) -> list[dict[str, Any]]:
        """Detect tool calls in response content.

        Args:
            content: Response content to parse.

        Returns:
            List of tool call dictionaries with 'name' and 'arguments' keys.
        """
        matches = self._TOOL_CALL_PATTERN.findall(content)

        tool_calls = []
        for name, args_str in matches:
            try:
                # Parse JSON arguments
                arguments = json.loads(args_str) if args_str.strip() else {}
                tool_calls.append({
                    "name": name,
                    "arguments": arguments,
                })
            except json.JSONDecodeError:
                # Invalid JSON, skip this tool call
                continue

        return tool_calls

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool from the registry.

        Args:
            tool_name: Name of tool to execute.
            arguments: Arguments to pass to tool.

        Returns:
            Tool execution result.

        Raises:
            MiddlewareError: If tool execution fails.
        """
        try:
            tool = self._registry.get(tool_name)
        except MiddlewareError as e:
            raise MiddlewareError(f"Tool '{tool_name}' not found") from e

        # Validate arguments
        tool.validate_arguments(arguments)

        # Execute tool
        try:
            result = tool.function(**arguments)
            return result
        except Exception as e:
            raise MiddlewareError(
                f"Tool '{tool_name}' execution failed: {e}"
            ) from e

    def _inject_tool_results(
        self,
        original_prompt: str,
        response_content: str,
        tool_results: list[dict[str, Any]],
    ) -> str:
        """Inject tool results back into conversation as new prompt.

        Args:
            original_prompt: Original user prompt.
            response_content: Model response with tool calls.
            tool_results: List of tool results with 'name' and 'result' keys.

        Returns:
            New prompt with conversation history and tool results.
        """
        # Build conversation history
        prompt_parts = [
            f"Previous conversation:",
            f"User: {original_prompt}",
            f"Assistant: {response_content}",
            "",
            "Tool results:",
        ]

        for result in tool_results:
            prompt_parts.append(
                f"- {result['name']}: {json.dumps(result['result'])}"
            )

        prompt_parts.append("")
        prompt_parts.append("Please continue based on the tool results above.")

        return "\n".join(prompt_parts)

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Send request with automatic tool calling.

        Args:
            request: The request to send.

        Returns:
            Final response after all tool calls resolved.

        Raises:
            MiddlewareError: If tool execution fails or max iterations exceeded.
        """
        current_request = request
        current_prompt = request.prompt

        for iteration in range(self._max_iterations):
            # Send request to wrapped client
            response = await self._client.send_request(current_request)

            # Detect tool calls in response
            tool_calls = self._detect_tool_calls(response.content)

            if not tool_calls:
                # No tool calls, return final response
                return response

            # Execute all detected tool calls
            tool_results = []
            for tool_call in tool_calls:
                result = self._execute_tool(
                    tool_call["name"],
                    tool_call["arguments"],
                )
                tool_results.append({
                    "name": tool_call["name"],
                    "result": result,
                })

            # Inject results back into conversation
            new_prompt = self._inject_tool_results(
                current_prompt,
                response.content,
                tool_results,
            )

            # Create new request for next iteration
            current_request = AgentRequest(
                prompt=new_prompt,
                system_prompt=request.system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop,
                attachments=request.attachments,
                metadata=request.metadata,
            )
            current_prompt = new_prompt

        # Max iterations reached
        raise MiddlewareError(
            f"Max iterations ({self._max_iterations}) reached with unresolved tool calls"
        )

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming request with automatic tool calling.

        Note: Streaming is not fully supported with function calling.
        Converts to non-streaming internally.

        Args:
            request: The request to send.

        Yields:
            Stream chunks from final response.
        """
        # For simplicity, convert to non-streaming
        # In production, would need to buffer and detect tool calls in stream
        response = await self.send_request(request)

        # Yield as single final chunk
        yield StreamChunk(
            delta=response.content,
            is_final=True,
            metadata={
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
