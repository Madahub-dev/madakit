"""Tests for prompt template middleware.

Tests PromptTemplateMiddleware constructor, template registration, variable
extraction, rendering, validation, and integration.
"""

from __future__ import annotations

import pytest

from mada_modelkit._errors import MiddlewareError
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk
from mada_modelkit.middleware.prompt_template import PromptTemplateMiddleware

from helpers import MockProvider


class TestModuleExports:
    """Test module-level exports and imports."""

    def test_all_exports(self) -> None:
        """__all__ contains only PromptTemplateMiddleware."""
        from mada_modelkit.middleware import prompt_template

        assert prompt_template.__all__ == ["PromptTemplateMiddleware"]

    def test_middleware_importable(self) -> None:
        """PromptTemplateMiddleware can be imported from module."""
        from mada_modelkit.middleware.prompt_template import (
            PromptTemplateMiddleware as PTM,
        )

        assert PTM is not None


class TestPromptTemplateMiddlewareConstructor:
    """Test PromptTemplateMiddleware constructor and initialization."""

    def test_minimal_constructor(self) -> None:
        """Constructor accepts client with no templates."""
        mock = MockProvider()

        middleware = PromptTemplateMiddleware(client=mock)

        assert middleware._client is mock
        assert middleware._templates == {}

    def test_with_templates(self) -> None:
        """Constructor accepts templates dictionary."""
        mock = MockProvider()
        templates = {
            "greeting": "Hello {{ name }}!",
            "question": "What is {{ topic }}?",
        }

        middleware = PromptTemplateMiddleware(client=mock, templates=templates)

        assert middleware._templates == templates

    def test_super_init_called(self) -> None:
        """PromptTemplateMiddleware calls BaseAgentClient.__init__."""
        mock = MockProvider()
        middleware = PromptTemplateMiddleware(client=mock)

        assert hasattr(middleware, "send_request")
        assert hasattr(middleware, "send_request_stream")


class TestTemplateRegistry:
    """Test template registration and retrieval."""

    def test_register_template(self) -> None:
        """register_template adds template to registry."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        middleware.register_template("test", "Hello {{ name }}")

        assert middleware._templates["test"] == "Hello {{ name }}"

    def test_register_multiple_templates(self) -> None:
        """Multiple templates can be registered."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        middleware.register_template("greeting", "Hi {{ name }}")
        middleware.register_template("farewell", "Bye {{ name }}")

        assert middleware._templates["greeting"] == "Hi {{ name }}"
        assert middleware._templates["farewell"] == "Bye {{ name }}"

    def test_register_overwrites_existing(self) -> None:
        """Registering same name overwrites existing template."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        middleware.register_template("test", "Version 1")
        middleware.register_template("test", "Version 2")

        assert middleware._templates["test"] == "Version 2"

    def test_get_template_success(self) -> None:
        """get_template retrieves registered template."""
        middleware = PromptTemplateMiddleware(
            client=MockProvider(),
            templates={"test": "Hello {{ name }}"},
        )

        template = middleware.get_template("test")

        assert template == "Hello {{ name }}"

    def test_get_template_not_found(self) -> None:
        """get_template raises MiddlewareError if not found."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        with pytest.raises(MiddlewareError, match="Template 'missing' not found"):
            middleware.get_template("missing")


class TestVariableExtraction:
    """Test variable extraction from templates."""

    def test_extract_single_variable(self) -> None:
        """Single variable is extracted."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        variables = middleware._extract_variables("Hello {{ name }}")

        assert variables == {"name"}

    def test_extract_multiple_variables(self) -> None:
        """Multiple variables are extracted."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        variables = middleware._extract_variables("{{ greeting }} {{ name }}, age {{ age }}")

        assert variables == {"greeting", "name", "age"}

    def test_extract_duplicate_variables(self) -> None:
        """Duplicate variables counted once."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        variables = middleware._extract_variables("{{ name }} and {{ name }}")

        assert variables == {"name"}

    def test_extract_with_whitespace(self) -> None:
        """Variables with whitespace around name are extracted."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        variables = middleware._extract_variables("{{  name  }} and {{ age}}")

        assert variables == {"name", "age"}

    def test_extract_no_variables(self) -> None:
        """Empty set when no variables."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        variables = middleware._extract_variables("Plain text with no variables")

        assert variables == set()

    def test_extract_underscore_variables(self) -> None:
        """Variables with underscores are extracted."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        variables = middleware._extract_variables("{{ user_name }} and {{ user_id }}")

        assert variables == {"user_name", "user_id"}


class TestVariableValidation:
    """Test variable validation."""

    def test_validate_all_provided(self) -> None:
        """Validation passes when all variables provided."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        # Should not raise
        middleware._validate_variables(
            "Hello {{ name }}, age {{ age }}",
            {"name": "Alice", "age": 30},
        )

    def test_validate_extra_variables_allowed(self) -> None:
        """Extra variables in dict are allowed."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        # Should not raise
        middleware._validate_variables(
            "Hello {{ name }}",
            {"name": "Alice", "age": 30, "city": "NYC"},
        )

    def test_validate_missing_single_variable(self) -> None:
        """Raises MiddlewareError when variable missing."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        with pytest.raises(MiddlewareError, match="Missing required variables: name"):
            middleware._validate_variables(
                "Hello {{ name }}",
                {},
            )

    def test_validate_missing_multiple_variables(self) -> None:
        """Raises MiddlewareError listing all missing variables."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        with pytest.raises(MiddlewareError, match="Missing required variables"):
            middleware._validate_variables(
                "{{ greeting }} {{ name }}, age {{ age }}",
                {"greeting": "Hello"},
            )

    def test_validate_no_variables_required(self) -> None:
        """Validation passes when template has no variables."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        # Should not raise
        middleware._validate_variables("Plain text", {})


class TestTemplateRendering:
    """Test template rendering with variable substitution."""

    def test_render_single_variable(self) -> None:
        """Renders template with single variable."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        result = middleware.render("Hello {{ name }}!", {"name": "Alice"})

        assert result == "Hello Alice!"

    def test_render_multiple_variables(self) -> None:
        """Renders template with multiple variables."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        result = middleware.render(
            "{{ greeting }} {{ name }}, you are {{ age }} years old",
            {"greeting": "Hi", "name": "Bob", "age": 25},
        )

        assert result == "Hi Bob, you are 25 years old"

    def test_render_repeated_variable(self) -> None:
        """Renders same variable multiple times."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        result = middleware.render(
            "{{ name }} and {{ name }} are friends",
            {"name": "Alice"},
        )

        assert result == "Alice and Alice are friends"

    def test_render_with_whitespace(self) -> None:
        """Renders variables with whitespace in placeholders."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        result = middleware.render(
            "{{  name  }} is {{ age}}",
            {"name": "Charlie", "age": 30},
        )

        assert result == "Charlie is 30"

    def test_render_no_variables(self) -> None:
        """Renders plain text unchanged."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        result = middleware.render("Plain text with no variables", {})

        assert result == "Plain text with no variables"

    def test_render_numeric_value(self) -> None:
        """Renders numeric values as strings."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        result = middleware.render("Count: {{ count }}", {"count": 42})

        assert result == "Count: 42"

    def test_render_missing_variable_raises(self) -> None:
        """Raises MiddlewareError if variable missing."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        with pytest.raises(MiddlewareError, match="Missing required variables"):
            middleware.render("Hello {{ name }}", {})

    def test_render_template_by_name(self) -> None:
        """render_template renders registered template."""
        middleware = PromptTemplateMiddleware(
            client=MockProvider(),
            templates={"greeting": "Hello {{ name }}!"},
        )

        result = middleware.render_template("greeting", {"name": "Dave"})

        assert result == "Hello Dave!"

    def test_render_template_not_found(self) -> None:
        """render_template raises if template not found."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        with pytest.raises(MiddlewareError, match="Template 'missing' not found"):
            middleware.render_template("missing", {})


class TestRequestTemplating:
    """Test template rendering in send_request."""

    @pytest.mark.asyncio
    async def test_send_request_no_templating(self) -> None:
        """send_request passes through when no templating requested."""
        calls = []

        class InspectProvider(MockProvider):
            async def send_request(self, request):
                calls.append(request.prompt)
                return await super().send_request(request)

        middleware = PromptTemplateMiddleware(client=InspectProvider())

        request = AgentRequest(prompt="Plain prompt")
        await middleware.send_request(request)

        assert calls == ["Plain prompt"]

    @pytest.mark.asyncio
    async def test_send_request_with_template_flag(self) -> None:
        """send_request renders prompt as template when metadata.template=True."""
        calls = []

        class InspectProvider(MockProvider):
            async def send_request(self, request):
                calls.append(request.prompt)
                return await super().send_request(request)

        middleware = PromptTemplateMiddleware(client=InspectProvider())

        request = AgentRequest(
            prompt="Hello {{ name }}!",
            metadata={"template": True, "variables": {"name": "Alice"}},
        )
        await middleware.send_request(request)

        assert calls == ["Hello Alice!"]

    @pytest.mark.asyncio
    async def test_send_request_with_template_name(self) -> None:
        """send_request renders named template."""
        calls = []

        class InspectProvider(MockProvider):
            async def send_request(self, request):
                calls.append(request.prompt)
                return await super().send_request(request)

        middleware = PromptTemplateMiddleware(
            client=InspectProvider(),
            templates={"greeting": "Hi {{ name }}, welcome to {{ place }}!"},
        )

        request = AgentRequest(
            prompt="",  # Ignored when template_name provided
            metadata={
                "template_name": "greeting",
                "variables": {"name": "Bob", "place": "NYC"},
            },
        )
        await middleware.send_request(request)

        assert calls == ["Hi Bob, welcome to NYC!"]

    @pytest.mark.asyncio
    async def test_send_request_preserves_other_fields(self) -> None:
        """send_request preserves request fields other than prompt."""
        calls = []

        class InspectProvider(MockProvider):
            async def send_request(self, request):
                calls.append((request.prompt, request.max_tokens, request.temperature))
                return await super().send_request(request)

        middleware = PromptTemplateMiddleware(client=InspectProvider())

        request = AgentRequest(
            prompt="Question: {{ topic }}",
            max_tokens=100,
            temperature=0.7,
            metadata={"template": True, "variables": {"topic": "AI"}},
        )
        await middleware.send_request(request)

        assert calls == [("Question: AI", 100, 0.7)]

    @pytest.mark.asyncio
    async def test_send_request_missing_variable_raises(self) -> None:
        """send_request raises MiddlewareError on missing variable."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        request = AgentRequest(
            prompt="Hello {{ name }}!",
            metadata={"template": True, "variables": {}},
        )

        with pytest.raises(MiddlewareError, match="Missing required variables"):
            await middleware.send_request(request)

    @pytest.mark.asyncio
    async def test_send_request_template_not_found_raises(self) -> None:
        """send_request raises MiddlewareError if template not found."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        request = AgentRequest(
            prompt="",
            metadata={"template_name": "missing", "variables": {}},
        )

        with pytest.raises(MiddlewareError, match="Template 'missing' not found"):
            await middleware.send_request(request)

    @pytest.mark.asyncio
    async def test_send_request_returns_response(self) -> None:
        """send_request returns response from wrapped client."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        request = AgentRequest(
            prompt="Hello {{ name }}",
            metadata={"template": True, "variables": {"name": "Test"}},
        )
        response = await middleware.send_request(request)

        assert response.content == "mock"
        assert response.model == "mock"


class TestStreamTemplating:
    """Test template rendering in send_request_stream."""

    @pytest.mark.asyncio
    async def test_send_request_stream_no_templating(self) -> None:
        """send_request_stream passes through when no templating."""
        calls = []

        class InspectProvider(MockProvider):
            async def send_request_stream(self, request):
                calls.append(request.prompt)
                yield StreamChunk(delta="test", is_final=True)

        middleware = PromptTemplateMiddleware(client=InspectProvider())

        request = AgentRequest(prompt="Plain prompt")
        chunks = []
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        assert calls == ["Plain prompt"]
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_send_request_stream_with_template(self) -> None:
        """send_request_stream renders template."""
        calls = []

        class InspectProvider(MockProvider):
            async def send_request_stream(self, request):
                calls.append(request.prompt)
                yield StreamChunk(delta="response", is_final=True)

        middleware = PromptTemplateMiddleware(client=InspectProvider())

        request = AgentRequest(
            prompt="Hello {{ name }}",
            metadata={"template": True, "variables": {"name": "Stream"}},
        )
        chunks = []
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        assert calls == ["Hello Stream"]

    @pytest.mark.asyncio
    async def test_send_request_stream_with_template_name(self) -> None:
        """send_request_stream renders named template."""
        calls = []

        class InspectProvider(MockProvider):
            async def send_request_stream(self, request):
                calls.append(request.prompt)
                yield StreamChunk(delta="test", is_final=True)

        middleware = PromptTemplateMiddleware(
            client=InspectProvider(),
            templates={"question": "What is {{ topic }}?"},
        )

        request = AgentRequest(
            prompt="",
            metadata={
                "template_name": "question",
                "variables": {"topic": "streaming"},
            },
        )
        async for _ in middleware.send_request_stream(request):
            pass

        assert calls == ["What is streaming?"]

    @pytest.mark.asyncio
    async def test_send_request_stream_preserves_chunks(self) -> None:
        """send_request_stream preserves all chunks."""

        class MultiChunkProvider(MockProvider):
            async def send_request_stream(self, request):
                yield StreamChunk(delta="a", is_final=False)
                yield StreamChunk(delta="b", is_final=False)
                yield StreamChunk(delta="c", is_final=True)

        middleware = PromptTemplateMiddleware(client=MultiChunkProvider())

        request = AgentRequest(
            prompt="Test {{ value }}",
            metadata={"template": True, "variables": {"value": "123"}},
        )
        deltas = []
        async for chunk in middleware.send_request_stream(request):
            deltas.append(chunk.delta)

        assert deltas == ["a", "b", "c"]


class TestIntegration:
    """Test full integration scenarios."""

    @pytest.mark.asyncio
    async def test_complex_template_rendering(self) -> None:
        """Complex template with multiple variables renders correctly."""
        middleware = PromptTemplateMiddleware(
            client=MockProvider(),
            templates={
                "analysis": "Analyze {{ subject }} focusing on {{ aspect1 }} and {{ aspect2 }}. "
                           "Context: {{ context }}"
            },
        )

        request = AgentRequest(
            prompt="",
            metadata={
                "template_name": "analysis",
                "variables": {
                    "subject": "AI safety",
                    "aspect1": "alignment",
                    "aspect2": "robustness",
                    "context": "production systems",
                },
            },
        )

        # We can't inspect the exact prompt sent, but verify no exception
        response = await middleware.send_request(request)
        assert response.content == "mock"

    @pytest.mark.asyncio
    async def test_template_priority(self) -> None:
        """template_name takes priority over template flag."""
        calls = []

        class InspectProvider(MockProvider):
            async def send_request(self, request):
                calls.append(request.prompt)
                return await super().send_request(request)

        middleware = PromptTemplateMiddleware(
            client=InspectProvider(),
            templates={"named": "Named: {{ var }}"},
        )

        request = AgentRequest(
            prompt="Inline: {{ var }}",
            metadata={
                "template": True,
                "template_name": "named",
                "variables": {"var": "test"},
            },
        )
        await middleware.send_request(request)

        # template_name should win
        assert calls == ["Named: test"]

    @pytest.mark.asyncio
    async def test_dynamic_template_registration(self) -> None:
        """Templates can be registered after initialization."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        # Register after creation
        middleware.register_template("dynamic", "Dynamic: {{ value }}")

        request = AgentRequest(
            prompt="",
            metadata={
                "template_name": "dynamic",
                "variables": {"value": "works"},
            },
        )

        # Should work without error
        response = await middleware.send_request(request)
        assert response is not None

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """PromptTemplateMiddleware works as context manager."""
        middleware = PromptTemplateMiddleware(client=MockProvider())

        async with middleware:
            request = AgentRequest(prompt="test")
            response = await middleware.send_request(request)
            assert response.content == "mock"
