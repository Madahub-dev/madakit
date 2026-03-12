"""Tool registry for function calling.

Register and manage callable tools with OpenAPI schema generation.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from mada_modelkit._errors import MiddlewareError

__all__ = ["Tool", "ToolRegistry"]


@dataclass
class Tool:
    """Tool definition for function calling.

    Represents a callable function with metadata and parameter schema.
    """

    name: str
    description: str
    function: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate tool definition."""
        if not self.name:
            raise ValueError("Tool name is required")
        if not self.description:
            raise ValueError("Tool description is required")
        if not callable(self.function):
            raise ValueError("Tool function must be callable")

        # Validate parameters schema if provided
        if self.parameters:
            if not isinstance(self.parameters, dict):
                raise ValueError("Tool parameters must be a dictionary")

            # Basic schema validation
            if "type" in self.parameters and self.parameters["type"] != "object":
                raise ValueError("Tool parameters type must be 'object'")

    def to_openapi_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAPI function schema.

        Returns:
            OpenAPI-compatible function schema.
        """
        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
            },
        }

        # Add parameters if provided
        if self.parameters:
            schema["function"]["parameters"] = self.parameters
        else:
            # Default to empty object schema
            schema["function"]["parameters"] = {
                "type": "object",
                "properties": {},
            }

        return schema

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Validate arguments against parameter schema.

        Args:
            arguments: Arguments to validate.

        Raises:
            MiddlewareError: If validation fails.
        """
        if not self.parameters:
            # No schema, accept any arguments
            return

        properties = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        # Check required parameters
        for param in required:
            if param not in arguments:
                raise MiddlewareError(
                    f"Missing required parameter '{param}' for tool '{self.name}'"
                )

        # Check parameter types (basic validation)
        for param_name, param_value in arguments.items():
            if param_name in properties:
                expected_type = properties[param_name].get("type")
                if expected_type:
                    actual_type = self._get_json_type(param_value)
                    if actual_type != expected_type:
                        raise MiddlewareError(
                            f"Parameter '{param_name}' for tool '{self.name}' "
                            f"expected type '{expected_type}', got '{actual_type}'"
                        )

    def _get_json_type(self, value: Any) -> str:
        """Get JSON schema type for a Python value.

        Args:
            value: Python value.

        Returns:
            JSON schema type string.
        """
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        elif value is None:
            return "null"
        else:
            return "string"  # Fallback


class ToolRegistry:
    """Registry for managing callable tools.

    Stores tools and provides registration, retrieval, and listing capabilities.
    """

    def __init__(self) -> None:
        """Initialise empty tool registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool to register.

        Raises:
            ValueError: If tool with same name already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        function: Callable[..., Any],
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Register a function as a tool.

        Args:
            name: Tool name.
            function: Callable function.
            description: Tool description.
            parameters: Optional parameter schema.
        """
        tool = Tool(
            name=name,
            description=description,
            function=function,
            parameters=parameters or {},
        )
        self.register(tool)

    def get(self, name: str) -> Tool:
        """Retrieve a tool by name.

        Args:
            name: Tool name.

        Returns:
            Tool instance.

        Raises:
            MiddlewareError: If tool not found.
        """
        if name not in self._tools:
            raise MiddlewareError(f"Tool '{name}' not found in registry")

        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        """List all registered tools.

        Returns:
            List of all tools in registry.
        """
        return list(self._tools.values())

    def to_openapi_schemas(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAPI function schemas.

        Returns:
            List of OpenAPI-compatible function schemas.
        """
        return [tool.to_openapi_schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools
