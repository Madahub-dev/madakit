"""Tests for tool registry.

Tests Tool dataclass, ToolRegistry registration, retrieval, schema generation,
and parameter validation.
"""

from __future__ import annotations

import pytest

from mada_modelkit._errors import MiddlewareError
from mada_modelkit.tools.registry import Tool, ToolRegistry


class TestModuleExports:
    """Test module-level exports and imports."""

    def test_all_exports(self) -> None:
        """__all__ contains Tool and ToolRegistry."""
        from mada_modelkit.tools import registry

        assert set(registry.__all__) == {"Tool", "ToolRegistry"}

    def test_classes_importable(self) -> None:
        """Tool and ToolRegistry can be imported from module."""
        from mada_modelkit.tools.registry import Tool as T, ToolRegistry as TR

        assert T is not None
        assert TR is not None


class TestToolDataclass:
    """Test Tool dataclass definition and validation."""

    def test_minimal_tool(self) -> None:
        """Tool can be created with required fields."""
        def example_func():
            pass

        tool = Tool(
            name="test_tool",
            description="Test description",
            function=example_func,
        )

        assert tool.name == "test_tool"
        assert tool.description == "Test description"
        assert tool.function is example_func
        assert tool.parameters == {}

    def test_tool_with_parameters(self) -> None:
        """Tool can be created with parameter schema."""
        def example_func(x: int):
            return x

        parameters = {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "Input value"},
            },
            "required": ["x"],
        }

        tool = Tool(
            name="increment",
            description="Increment a number",
            function=example_func,
            parameters=parameters,
        )

        assert tool.parameters == parameters

    def test_empty_name_raises(self) -> None:
        """Empty tool name raises ValueError."""
        with pytest.raises(ValueError, match="Tool name is required"):
            Tool(name="", description="test", function=lambda: None)

    def test_empty_description_raises(self) -> None:
        """Empty tool description raises ValueError."""
        with pytest.raises(ValueError, match="Tool description is required"):
            Tool(name="test", description="", function=lambda: None)

    def test_non_callable_function_raises(self) -> None:
        """Non-callable function raises ValueError."""
        with pytest.raises(ValueError, match="Tool function must be callable"):
            Tool(name="test", description="test", function="not_callable")

    def test_invalid_parameters_type_raises(self) -> None:
        """Non-dict parameters raises ValueError."""
        with pytest.raises(ValueError, match="Tool parameters must be a dictionary"):
            Tool(
                name="test",
                description="test",
                function=lambda: None,
                parameters="invalid",
            )

    def test_invalid_parameters_schema_raises(self) -> None:
        """Parameters with non-object type raises ValueError."""
        with pytest.raises(ValueError, match="Tool parameters type must be 'object'"):
            Tool(
                name="test",
                description="test",
                function=lambda: None,
                parameters={"type": "string"},
            )


class TestOpenAPISchemaGeneration:
    """Test OpenAPI schema generation."""

    def test_basic_schema(self) -> None:
        """Basic tool generates valid OpenAPI schema."""
        def greet():
            return "Hello"

        tool = Tool(
            name="greet",
            description="Greet the user",
            function=greet,
        )

        schema = tool.to_openapi_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "greet"
        assert schema["function"]["description"] == "Greet the user"
        assert schema["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }

    def test_schema_with_parameters(self) -> None:
        """Tool with parameters generates schema with parameters."""
        def add(a: int, b: int) -> int:
            return a + b

        parameters = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        }

        tool = Tool(
            name="add",
            description="Add two numbers",
            function=add,
            parameters=parameters,
        )

        schema = tool.to_openapi_schema()

        assert schema["function"]["parameters"] == parameters

    def test_schema_structure(self) -> None:
        """Schema has correct structure."""
        tool = Tool(name="test", description="Test", function=lambda: None)

        schema = tool.to_openapi_schema()

        assert "type" in schema
        assert "function" in schema
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


class TestParameterValidation:
    """Test parameter validation against schema."""

    def test_validate_no_schema(self) -> None:
        """Validation passes when no schema defined."""
        tool = Tool(name="test", description="Test", function=lambda: None)

        # Should not raise
        tool.validate_arguments({"any": "value"})
        tool.validate_arguments({})

    def test_validate_required_parameters(self) -> None:
        """Validation checks required parameters present."""
        parameters = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }

        tool = Tool(
            name="greet",
            description="Greet by name",
            function=lambda name: f"Hello {name}",
            parameters=parameters,
        )

        # Should not raise
        tool.validate_arguments({"name": "Alice"})

        # Should raise for missing required parameter
        with pytest.raises(MiddlewareError, match="Missing required parameter 'name'"):
            tool.validate_arguments({})

    def test_validate_parameter_types(self) -> None:
        """Validation checks parameter types."""
        parameters = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
        }

        tool = Tool(
            name="repeat",
            description="Repeat count times",
            function=lambda count: "x" * count,
            parameters=parameters,
        )

        # Should not raise for correct type
        tool.validate_arguments({"count": 5})

        # Should raise for wrong type
        with pytest.raises(MiddlewareError, match="expected type 'integer'"):
            tool.validate_arguments({"count": "five"})

    def test_validate_multiple_types(self) -> None:
        """Validation handles multiple parameter types."""
        parameters = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
            },
            "required": ["name"],
        }

        tool = Tool(
            name="create_user",
            description="Create a user",
            function=lambda **kwargs: kwargs,
            parameters=parameters,
        )

        # Valid arguments
        tool.validate_arguments({"name": "Bob", "age": 30, "active": True})

        # Invalid type for age
        with pytest.raises(MiddlewareError, match="expected type 'integer'"):
            tool.validate_arguments({"name": "Bob", "age": "30"})

    def test_get_json_type(self) -> None:
        """_get_json_type correctly maps Python types to JSON types."""
        tool = Tool(name="test", description="Test", function=lambda: None)

        assert tool._get_json_type(True) == "boolean"
        assert tool._get_json_type(42) == "integer"
        assert tool._get_json_type(3.14) == "number"
        assert tool._get_json_type("hello") == "string"
        assert tool._get_json_type([1, 2, 3]) == "array"
        assert tool._get_json_type({"key": "value"}) == "object"
        assert tool._get_json_type(None) == "null"


class TestToolRegistryBasics:
    """Test ToolRegistry basic functionality."""

    def test_empty_registry(self) -> None:
        """Empty registry is initialized correctly."""
        registry = ToolRegistry()

        assert len(registry) == 0
        assert registry.list_tools() == []

    def test_register_tool(self) -> None:
        """Tool can be registered."""
        registry = ToolRegistry()
        tool = Tool(name="test", description="Test", function=lambda: None)

        registry.register(tool)

        assert len(registry) == 1
        assert "test" in registry

    def test_register_duplicate_raises(self) -> None:
        """Registering duplicate tool name raises ValueError."""
        registry = ToolRegistry()
        tool1 = Tool(name="test", description="Test 1", function=lambda: 1)
        tool2 = Tool(name="test", description="Test 2", function=lambda: 2)

        registry.register(tool1)

        with pytest.raises(ValueError, match="Tool 'test' already registered"):
            registry.register(tool2)

    def test_register_function(self) -> None:
        """Function can be registered directly."""
        registry = ToolRegistry()

        def greet(name: str) -> str:
            return f"Hello {name}"

        registry.register_function(
            name="greet",
            function=greet,
            description="Greet by name",
        )

        assert len(registry) == 1
        assert "greet" in registry

        tool = registry.get("greet")
        assert tool.name == "greet"
        assert tool.function is greet

    def test_register_function_with_parameters(self) -> None:
        """Function with parameters can be registered."""
        registry = ToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        parameters = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
        }

        registry.register_function(
            name="add",
            function=add,
            description="Add numbers",
            parameters=parameters,
        )

        tool = registry.get("add")
        assert tool.parameters == parameters


class TestToolRetrieval:
    """Test tool retrieval from registry."""

    def test_get_existing_tool(self) -> None:
        """get() retrieves registered tool."""
        registry = ToolRegistry()
        tool = Tool(name="test", description="Test", function=lambda: None)
        registry.register(tool)

        retrieved = registry.get("test")

        assert retrieved is tool

    def test_get_nonexistent_tool_raises(self) -> None:
        """get() raises MiddlewareError for nonexistent tool."""
        registry = ToolRegistry()

        with pytest.raises(MiddlewareError, match="Tool 'missing' not found"):
            registry.get("missing")

    def test_list_tools(self) -> None:
        """list_tools() returns all registered tools."""
        registry = ToolRegistry()
        tool1 = Tool(name="tool1", description="Test 1", function=lambda: 1)
        tool2 = Tool(name="tool2", description="Test 2", function=lambda: 2)

        registry.register(tool1)
        registry.register(tool2)

        tools = registry.list_tools()

        assert len(tools) == 2
        assert tool1 in tools
        assert tool2 in tools

    def test_contains(self) -> None:
        """__contains__ checks tool existence."""
        registry = ToolRegistry()
        tool = Tool(name="test", description="Test", function=lambda: None)

        assert "test" not in registry

        registry.register(tool)

        assert "test" in registry
        assert "other" not in registry


class TestRegistrySchemaGeneration:
    """Test registry-wide schema generation."""

    def test_to_openapi_schemas(self) -> None:
        """to_openapi_schemas() generates schemas for all tools."""
        registry = ToolRegistry()

        registry.register_function(
            name="greet",
            function=lambda: "Hello",
            description="Greet user",
        )
        registry.register_function(
            name="add",
            function=lambda a, b: a + b,
            description="Add numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
            },
        )

        schemas = registry.to_openapi_schemas()

        assert len(schemas) == 2
        assert all(s["type"] == "function" for s in schemas)
        assert any(s["function"]["name"] == "greet" for s in schemas)
        assert any(s["function"]["name"] == "add" for s in schemas)

    def test_empty_registry_schemas(self) -> None:
        """Empty registry returns empty schema list."""
        registry = ToolRegistry()

        schemas = registry.to_openapi_schemas()

        assert schemas == []


class TestIntegration:
    """Test integration scenarios."""

    def test_full_workflow(self) -> None:
        """Full registration, retrieval, and execution workflow."""
        registry = ToolRegistry()

        def calculate(operation: str, a: int, b: int) -> int:
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            return 0

        parameters = {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["operation", "a", "b"],
        }

        registry.register_function(
            name="calculate",
            function=calculate,
            description="Perform calculation",
            parameters=parameters,
        )

        # Retrieve and validate
        tool = registry.get("calculate")
        tool.validate_arguments({"operation": "add", "a": 5, "b": 3})

        # Execute
        result = tool.function(operation="add", a=5, b=3)
        assert result == 8

        # Generate schema
        schema = tool.to_openapi_schema()
        assert schema["function"]["name"] == "calculate"

    def test_multiple_tools_registration(self) -> None:
        """Multiple tools can be registered and managed."""
        registry = ToolRegistry()

        # Register multiple tools
        for i in range(5):
            registry.register_function(
                name=f"tool_{i}",
                function=lambda x=i: x,
                description=f"Tool {i}",
            )

        assert len(registry) == 5

        # All tools retrievable
        for i in range(5):
            tool = registry.get(f"tool_{i}")
            assert tool.name == f"tool_{i}"

        # Schemas generated for all
        schemas = registry.to_openapi_schemas()
        assert len(schemas) == 5
