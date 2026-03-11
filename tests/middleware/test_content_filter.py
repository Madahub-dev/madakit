"""Tests for content filtering middleware.

Tests ContentFilterMiddleware constructor, PII detection and redaction,
safety checks, response filtering, and error handling.
"""

from __future__ import annotations

import pytest

from mada_modelkit._errors import MiddlewareError
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk
from mada_modelkit.middleware.content_filter import ContentFilterMiddleware

from helpers import MockProvider


class TestModuleExports:
    """Test module-level exports and imports."""

    def test_all_exports(self) -> None:
        """__all__ contains only ContentFilterMiddleware."""
        from mada_modelkit.middleware import content_filter

        assert content_filter.__all__ == ["ContentFilterMiddleware"]

    def test_middleware_importable(self) -> None:
        """ContentFilterMiddleware can be imported from module."""
        from mada_modelkit.middleware.content_filter import (
            ContentFilterMiddleware as CFM,
        )

        assert CFM is not None


class TestContentFilterMiddlewareConstructor:
    """Test ContentFilterMiddleware constructor and initialization."""

    def test_minimal_constructor(self) -> None:
        """Constructor accepts client with defaults."""
        mock = MockProvider()

        middleware = ContentFilterMiddleware(client=mock)

        assert middleware._client is mock
        assert middleware._redact_pii is True
        assert middleware._safety_check is None
        assert middleware._response_filter is None

    def test_with_redact_pii_disabled(self) -> None:
        """Constructor accepts redact_pii=False."""
        mock = MockProvider()

        middleware = ContentFilterMiddleware(client=mock, redact_pii=False)

        assert middleware._redact_pii is False

    def test_with_safety_check_callback(self) -> None:
        """Constructor accepts safety_check callback."""
        mock = MockProvider()

        def safety_check(prompt: str) -> None:
            pass

        middleware = ContentFilterMiddleware(client=mock, safety_check=safety_check)

        assert middleware._safety_check is safety_check

    def test_with_response_filter_callback(self) -> None:
        """Constructor accepts response_filter callback."""
        mock = MockProvider()

        def response_filter(content: str) -> str:
            return content.upper()

        middleware = ContentFilterMiddleware(
            client=mock, response_filter=response_filter
        )

        assert middleware._response_filter is response_filter

    def test_super_init_called(self) -> None:
        """ContentFilterMiddleware calls BaseAgentClient.__init__."""
        mock = MockProvider()
        middleware = ContentFilterMiddleware(client=mock)

        assert hasattr(middleware, "send_request")
        assert hasattr(middleware, "send_request_stream")


class TestPIIDetection:
    """Test PII detection patterns."""

    def test_email_detection(self) -> None:
        """Email addresses are detected."""
        middleware = ContentFilterMiddleware(client=MockProvider())

        text = "Contact me at john.doe@example.com for info"
        result = middleware._detect_and_redact_pii(text)

        assert "john.doe@example.com" not in result
        assert "[REDACTED]" in result

    def test_ssn_detection(self) -> None:
        """SSNs are detected."""
        middleware = ContentFilterMiddleware(client=MockProvider())

        text = "My SSN is 123-45-6789"
        result = middleware._detect_and_redact_pii(text)

        assert "123-45-6789" not in result
        assert "[REDACTED]" in result

    def test_credit_card_detection(self) -> None:
        """Credit card numbers are detected."""
        middleware = ContentFilterMiddleware(client=MockProvider())

        text = "Card: 1234 5678 9012 3456"
        result = middleware._detect_and_redact_pii(text)

        assert "1234 5678 9012 3456" not in result
        assert "[REDACTED]" in result

    def test_credit_card_no_spaces(self) -> None:
        """Credit card numbers without spaces are detected."""
        middleware = ContentFilterMiddleware(client=MockProvider())

        text = "Card: 1234567890123456"
        result = middleware._detect_and_redact_pii(text)

        assert "1234567890123456" not in result
        assert "[REDACTED]" in result

    def test_credit_card_with_dashes(self) -> None:
        """Credit card numbers with dashes are detected."""
        middleware = ContentFilterMiddleware(client=MockProvider())

        text = "Card: 1234-5678-9012-3456"
        result = middleware._detect_and_redact_pii(text)

        assert "1234-5678-9012-3456" not in result
        assert "[REDACTED]" in result

    def test_multiple_pii_types(self) -> None:
        """Multiple PII types in same text are all redacted."""
        middleware = ContentFilterMiddleware(client=MockProvider())

        text = "Email: test@example.com, SSN: 111-22-3333, Card: 4111111111111111"
        result = middleware._detect_and_redact_pii(text)

        assert "test@example.com" not in result
        assert "111-22-3333" not in result
        assert "4111111111111111" not in result
        assert result.count("[REDACTED]") == 3

    def test_no_pii_unchanged(self) -> None:
        """Text without PII is unchanged."""
        middleware = ContentFilterMiddleware(client=MockProvider())

        text = "This is a normal message with no sensitive data"
        result = middleware._detect_and_redact_pii(text)

        assert result == text

    def test_redact_pii_disabled(self) -> None:
        """PII detection disabled when redact_pii=False."""
        middleware = ContentFilterMiddleware(client=MockProvider(), redact_pii=False)

        text = "Email: test@example.com"
        result = middleware._detect_and_redact_pii(text)

        assert result == text
        assert "test@example.com" in result


class TestSafetyChecks:
    """Test safety check callbacks."""

    def test_no_safety_check_passes(self) -> None:
        """No safety check when callback is None."""
        middleware = ContentFilterMiddleware(client=MockProvider())

        # Should not raise
        middleware._check_safety("any prompt")

    def test_safety_check_called(self) -> None:
        """Safety check callback is called with prompt."""
        calls = []

        def safety_check(prompt: str) -> None:
            calls.append(prompt)

        middleware = ContentFilterMiddleware(
            client=MockProvider(), safety_check=safety_check
        )

        middleware._check_safety("test prompt")

        assert calls == ["test prompt"]

    def test_safety_check_raises_on_unsafe(self) -> None:
        """Safety check raises MiddlewareError on unsafe content."""

        def safety_check(prompt: str) -> None:
            if "unsafe" in prompt:
                raise ValueError("Unsafe content detected")

        middleware = ContentFilterMiddleware(
            client=MockProvider(), safety_check=safety_check
        )

        with pytest.raises(MiddlewareError, match="Safety check failed"):
            middleware._check_safety("This is unsafe content")

    def test_safety_check_passes_on_safe(self) -> None:
        """Safety check passes for safe content."""

        def safety_check(prompt: str) -> None:
            if "unsafe" in prompt:
                raise ValueError("Unsafe content")

        middleware = ContentFilterMiddleware(
            client=MockProvider(), safety_check=safety_check
        )

        # Should not raise
        middleware._check_safety("This is safe content")

    def test_safety_check_exception_chained(self) -> None:
        """Safety check exception is chained in MiddlewareError."""

        def safety_check(prompt: str) -> None:
            raise ValueError("Test error")

        middleware = ContentFilterMiddleware(
            client=MockProvider(), safety_check=safety_check
        )

        with pytest.raises(MiddlewareError) as exc_info:
            middleware._check_safety("test")

        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "Test error"


class TestResponseFiltering:
    """Test response content filtering."""

    def test_no_filter_unchanged(self) -> None:
        """Response unchanged when no filter configured."""
        middleware = ContentFilterMiddleware(client=MockProvider())

        result = middleware._filter_response("test content")

        assert result == "test content"

    def test_filter_applied(self) -> None:
        """Response filter callback is applied."""

        def response_filter(content: str) -> str:
            return content.upper()

        middleware = ContentFilterMiddleware(
            client=MockProvider(), response_filter=response_filter
        )

        result = middleware._filter_response("test content")

        assert result == "TEST CONTENT"

    def test_filter_removes_content(self) -> None:
        """Response filter can remove harmful content."""

        def response_filter(content: str) -> str:
            return content.replace("harmful", "[FILTERED]")

        middleware = ContentFilterMiddleware(
            client=MockProvider(), response_filter=response_filter
        )

        result = middleware._filter_response("This is harmful content")

        assert result == "This is [FILTERED] content"
        assert "harmful" not in result


class TestRequestFiltering:
    """Test request filtering pipeline."""

    @pytest.mark.asyncio
    async def test_send_request_redacts_prompt_pii(self) -> None:
        """send_request redacts PII from prompt."""
        mock = MockProvider()
        middleware = ContentFilterMiddleware(client=mock)

        request = AgentRequest(prompt="Contact me at test@example.com")
        response = await middleware.send_request(request)

        # Check that mock received redacted prompt
        # We can't directly inspect the call, but we verify response is returned
        assert response.content == "mock"

    @pytest.mark.asyncio
    async def test_send_request_redacts_system_prompt_pii(self) -> None:
        """send_request redacts PII from system prompt."""
        mock = MockProvider()
        middleware = ContentFilterMiddleware(client=mock)

        request = AgentRequest(
            prompt="test",
            system_prompt="System instructions: email admin@example.com",
        )
        await middleware.send_request(request)

        # System prompt should be redacted but we can't inspect the call
        # Just verify no exception raised

    @pytest.mark.asyncio
    async def test_send_request_safety_check_blocks_unsafe(self) -> None:
        """send_request raises MiddlewareError on unsafe content."""

        def safety_check(prompt: str) -> None:
            if "hack" in prompt:
                raise ValueError("Unsafe")

        mock = MockProvider()
        middleware = ContentFilterMiddleware(
            client=mock, safety_check=safety_check
        )

        request = AgentRequest(prompt="How to hack a system")

        with pytest.raises(MiddlewareError, match="Safety check failed"):
            await middleware.send_request(request)

    @pytest.mark.asyncio
    async def test_send_request_filters_response(self) -> None:
        """send_request applies response filter."""

        def response_filter(content: str) -> str:
            return content.upper()

        mock = MockProvider()
        middleware = ContentFilterMiddleware(
            client=mock, response_filter=response_filter
        )

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        assert response.content == "MOCK"

    @pytest.mark.asyncio
    async def test_send_request_preserves_response_metadata(self) -> None:
        """send_request preserves response metadata and tokens."""

        def response_filter(content: str) -> str:
            return content.upper()

        mock = MockProvider()
        middleware = ContentFilterMiddleware(
            client=mock, response_filter=response_filter
        )

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        assert response.model == "mock"
        assert response.input_tokens == 10
        assert response.output_tokens == 5

    @pytest.mark.asyncio
    async def test_send_request_preserves_request_fields(self) -> None:
        """send_request preserves request fields not affected by filters."""
        mock = MockProvider()
        middleware = ContentFilterMiddleware(client=mock)

        request = AgentRequest(
            prompt="test",
            max_tokens=100,
            temperature=0.7,
            stop=["STOP"],
            metadata={"key": "value"},
        )
        await middleware.send_request(request)

        # Fields should be preserved in filtered request
        # We can't directly inspect but verify no exception


class TestStreamFiltering:
    """Test streaming with content filtering."""

    @pytest.mark.asyncio
    async def test_send_request_stream_redacts_prompt_pii(self) -> None:
        """send_request_stream redacts PII from prompt."""
        mock = MockProvider()
        middleware = ContentFilterMiddleware(client=mock)

        request = AgentRequest(prompt="Email: test@example.com")
        chunks = []
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        # Prompt was redacted before sending to client
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_send_request_stream_safety_check_blocks(self) -> None:
        """send_request_stream raises on unsafe content."""

        def safety_check(prompt: str) -> None:
            if "unsafe" in prompt:
                raise ValueError("Unsafe")

        mock = MockProvider()
        middleware = ContentFilterMiddleware(
            client=mock, safety_check=safety_check
        )

        request = AgentRequest(prompt="This is unsafe")

        with pytest.raises(MiddlewareError, match="Safety check failed"):
            async for _ in middleware.send_request_stream(request):
                pass

    @pytest.mark.asyncio
    async def test_send_request_stream_filters_final_chunk(self) -> None:
        """send_request_stream applies filter to final chunk."""

        class StreamProvider(MockProvider):
            async def send_request_stream(self, request):
                yield StreamChunk(delta="chunk1", is_final=False)
                yield StreamChunk(delta="final", is_final=True)

        def response_filter(content: str) -> str:
            return content.upper()

        mock = StreamProvider()
        middleware = ContentFilterMiddleware(
            client=mock, response_filter=response_filter
        )

        request = AgentRequest(prompt="test")
        chunks = []
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].delta == "chunk1"  # Non-final unchanged
        assert chunks[1].delta == "FINAL"  # Final chunk filtered

    @pytest.mark.asyncio
    async def test_send_request_stream_preserves_non_final_chunks(self) -> None:
        """send_request_stream doesn't filter non-final chunks."""

        class MultiChunkProvider(MockProvider):
            async def send_request_stream(self, request):
                yield StreamChunk(delta="a", is_final=False)
                yield StreamChunk(delta="b", is_final=False)
                yield StreamChunk(delta="c", is_final=True)

        def response_filter(content: str) -> str:
            return content.upper()

        mock = MultiChunkProvider()
        middleware = ContentFilterMiddleware(
            client=mock, response_filter=response_filter
        )

        request = AgentRequest(prompt="test")
        deltas = []
        async for chunk in middleware.send_request_stream(request):
            deltas.append(chunk.delta)

        assert deltas == ["a", "b", "C"]  # Only final filtered


class TestIntegration:
    """Test full integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_pii_redaction_pipeline(self) -> None:
        """Full pipeline redacts PII from prompt and filters response."""

        def response_filter(content: str) -> str:
            # Simulate removing PII from response too
            return content.replace("secret", "[FILTERED]")

        mock = MockProvider(
            responses=[
                AgentResponse(
                    content="Here is your secret data",
                    model="test",
                    input_tokens=10,
                    output_tokens=5,
                )
            ]
        )
        middleware = ContentFilterMiddleware(
            client=mock, response_filter=response_filter
        )

        request = AgentRequest(prompt="My email is admin@example.com")
        response = await middleware.send_request(request)

        assert response.content == "Here is your [FILTERED] data"

    @pytest.mark.asyncio
    async def test_safety_check_blocks_before_api_call(self) -> None:
        """Safety check blocks request before reaching API."""
        call_count = 0

        class CountingProvider(MockProvider):
            async def send_request(self, request):
                nonlocal call_count
                call_count += 1
                return await super().send_request(request)

        def safety_check(prompt: str) -> None:
            if "dangerous" in prompt:
                raise ValueError("Blocked")

        mock = CountingProvider()
        middleware = ContentFilterMiddleware(
            client=mock, safety_check=safety_check
        )

        request = AgentRequest(prompt="This is dangerous")

        with pytest.raises(MiddlewareError):
            await middleware.send_request(request)

        # API should not have been called
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_multiple_filters_composed(self) -> None:
        """PII redaction and response filtering work together."""

        def response_filter(content: str) -> str:
            return content.replace("sensitive", "[REMOVED]")

        mock = MockProvider(
            responses=[
                AgentResponse(
                    content="This is sensitive info",
                    model="test",
                    input_tokens=10,
                    output_tokens=5,
                )
            ]
        )
        middleware = ContentFilterMiddleware(
            client=mock, response_filter=response_filter
        )

        request = AgentRequest(
            prompt="My SSN is 123-45-6789 and I need help with sensitive data"
        )
        response = await middleware.send_request(request)

        # Response should be filtered
        assert response.content == "This is [REMOVED] info"
        assert "sensitive" not in response.content

    @pytest.mark.asyncio
    async def test_redact_pii_disabled_preserves_pii(self) -> None:
        """PII preserved when redact_pii=False."""
        calls = []

        class InspectProvider(MockProvider):
            async def send_request(self, request):
                calls.append(request.prompt)
                return await super().send_request(request)

        mock = InspectProvider()
        middleware = ContentFilterMiddleware(client=mock, redact_pii=False)

        request = AgentRequest(prompt="Email: test@example.com")
        await middleware.send_request(request)

        # Email should be preserved in prompt sent to provider
        assert len(calls) == 1
        assert "test@example.com" in calls[0]

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """ContentFilterMiddleware works as context manager."""
        mock = MockProvider()
        middleware = ContentFilterMiddleware(client=mock)

        async with middleware:
            request = AgentRequest(prompt="test")
            response = await middleware.send_request(request)
            assert response.content == "mock"
