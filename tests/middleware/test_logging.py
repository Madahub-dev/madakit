"""Tests for logging middleware.

Tests LoggingMiddleware constructor, request/response/error logging,
correlation ID generation and propagation, and PII filtering.
"""

from __future__ import annotations

import logging
import pytest

from mada_modelkit.middleware.logging import LoggingMiddleware
from mada_modelkit._types import AgentRequest, AgentResponse

from helpers import MockProvider


class TestModuleExports:
    """Test module-level exports and imports."""

    def test_all_exports(self) -> None:
        """__all__ contains only LoggingMiddleware."""
        from mada_modelkit.middleware import logging as logging_module

        assert logging_module.__all__ == ["LoggingMiddleware"]

    def test_middleware_importable(self) -> None:
        """LoggingMiddleware can be imported from module."""
        from mada_modelkit.middleware.logging import LoggingMiddleware as LM

        assert LM is not None


class TestLoggingMiddlewareConstructor:
    """Test LoggingMiddleware constructor and initialization."""

    def test_minimal_constructor(self) -> None:
        """Constructor accepts client and uses defaults."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock)

        assert middleware._client is mock
        assert middleware._logger is not None
        assert middleware._log_level == logging.INFO
        assert middleware._include_prompts is False

    def test_explicit_logger(self) -> None:
        """Constructor accepts explicit logger instance."""
        mock = MockProvider()
        custom_logger = logging.getLogger("custom")
        middleware = LoggingMiddleware(client=mock, logger=custom_logger)

        assert middleware._logger is custom_logger

    def test_explicit_log_level(self) -> None:
        """Constructor accepts explicit log level."""
        mock = MockProvider()

        # Different log levels
        mw_debug = LoggingMiddleware(client=mock, log_level="DEBUG")
        assert mw_debug._log_level == logging.DEBUG

        mw_warning = LoggingMiddleware(client=mock, log_level="WARNING")
        assert mw_warning._log_level == logging.WARNING

        mw_error = LoggingMiddleware(client=mock, log_level="ERROR")
        assert mw_error._log_level == logging.ERROR

    def test_log_level_case_insensitive(self) -> None:
        """Log level string is case-insensitive."""
        mock = MockProvider()

        mw_lower = LoggingMiddleware(client=mock, log_level="info")
        assert mw_lower._log_level == logging.INFO

        mw_upper = LoggingMiddleware(client=mock, log_level="INFO")
        assert mw_upper._log_level == logging.INFO

        mw_mixed = LoggingMiddleware(client=mock, log_level="Info")
        assert mw_mixed._log_level == logging.INFO

    def test_include_prompts_flag(self) -> None:
        """Constructor accepts include_prompts flag."""
        mock = MockProvider()

        mw_excluded = LoggingMiddleware(client=mock, include_prompts=False)
        assert mw_excluded._include_prompts is False

        mw_included = LoggingMiddleware(client=mock, include_prompts=True)
        assert mw_included._include_prompts is True

    def test_default_logger_namespace(self) -> None:
        """Default logger uses correct namespace."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock)

        # Default logger should be from the logging module
        assert middleware._logger.name == "mada_modelkit.middleware.logging"

    def test_wraps_base_agent_client(self) -> None:
        """LoggingMiddleware can wrap any BaseAgentClient."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        mw1 = LoggingMiddleware(client=mock1)
        mw2 = LoggingMiddleware(client=mock2)

        assert mw1._client is mock1
        assert mw2._client is mock2

    def test_wraps_middleware(self) -> None:
        """LoggingMiddleware can wrap another middleware."""
        from mada_modelkit.middleware.retry import RetryMiddleware

        mock = MockProvider()
        retry_mw = RetryMiddleware(client=mock, max_retries=3)
        logging_mw = LoggingMiddleware(client=retry_mw, log_level="DEBUG")

        assert logging_mw._client is retry_mw
        assert logging_mw._log_level == logging.DEBUG

    def test_super_init_called(self) -> None:
        """LoggingMiddleware calls BaseAgentClient.__init__."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock)

        # Should have BaseAgentClient methods
        assert hasattr(middleware, "send_request")
        assert hasattr(middleware, "send_request_stream")
        assert hasattr(middleware, "cancel")
        assert hasattr(middleware, "close")

    def test_instance_isolation(self) -> None:
        """Different instances have independent state."""
        mock = MockProvider()
        logger1 = logging.getLogger("logger1")
        logger2 = logging.getLogger("logger2")

        mw1 = LoggingMiddleware(client=mock, logger=logger1, log_level="DEBUG")
        mw2 = LoggingMiddleware(client=mock, logger=logger2, log_level="ERROR")

        assert mw1._logger is logger1
        assert mw2._logger is logger2
        assert mw1._log_level == logging.DEBUG
        assert mw2._log_level == logging.ERROR


class TestRequestLogging:
    """Test request logging with IDs and metadata."""

    @pytest.mark.asyncio
    async def test_send_request_logs_request_start(self, caplog) -> None:
        """send_request logs request start with ID."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        await middleware.send_request(request)

        # Should have logged request start
        assert len(caplog.records) >= 1
        assert "Request started" in caplog.text
        assert "request_id" in caplog.records[0].__dict__

    @pytest.mark.asyncio
    async def test_request_id_is_unique(self, caplog) -> None:
        """Each request gets a unique ID."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        await middleware.send_request(request)
        first_id = caplog.records[0].__dict__.get("request_id")

        caplog.clear()
        await middleware.send_request(request)
        second_id = caplog.records[0].__dict__.get("request_id")

        assert first_id != second_id

    @pytest.mark.asyncio
    async def test_request_logging_includes_metadata(self, caplog) -> None:
        """Request log includes max_tokens, temperature, and metadata."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(
            prompt="test",
            max_tokens=512,
            temperature=0.9,
            metadata={"user_id": "user123", "session": "abc"},
        )

        await middleware.send_request(request)

        log_record = caplog.records[0]
        assert log_record.__dict__["max_tokens"] == 512
        assert log_record.__dict__["temperature"] == 0.9
        assert log_record.__dict__["metadata"] == {"user_id": "user123", "session": "abc"}

    @pytest.mark.asyncio
    async def test_prompt_excluded_by_default(self, caplog) -> None:
        """Prompt text is excluded from logs by default."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="secret prompt", system_prompt="secret system")

        await middleware.send_request(request)

        log_record = caplog.records[0]
        assert "prompt" not in log_record.__dict__
        assert "system_prompt" not in log_record.__dict__

    @pytest.mark.asyncio
    async def test_prompt_included_when_enabled(self, caplog) -> None:
        """Prompt text is included when include_prompts=True."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO", include_prompts=True)

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test prompt", system_prompt="test system")

        await middleware.send_request(request)

        log_record = caplog.records[0]
        assert log_record.__dict__["prompt"] == "test prompt"
        assert log_record.__dict__["system_prompt"] == "test system"

    @pytest.mark.asyncio
    async def test_send_request_stream_logs_request_start(self, caplog) -> None:
        """send_request_stream logs request start with ID."""
        class MultiChunkProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="chunk", is_final=True)

        mock = MultiChunkProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        async for _ in middleware.send_request_stream(request):
            pass

        # Should have logged request start
        assert len(caplog.records) >= 1
        assert "Request started" in caplog.text
        assert "request_id" in caplog.records[0].__dict__

    @pytest.mark.asyncio
    async def test_log_level_filters_logs(self, caplog) -> None:
        """Log level setting filters out lower-level logs."""
        mock = MockProvider()

        # WARNING level should not log INFO messages
        middleware = LoggingMiddleware(client=mock, log_level="WARNING")

        caplog.set_level(logging.WARNING)
        request = AgentRequest(prompt="test")

        await middleware.send_request(request)

        # No logs because WARNING > INFO
        assert len(caplog.records) == 0

    @pytest.mark.asyncio
    async def test_custom_logger_receives_logs(self, caplog) -> None:
        """Custom logger instance receives log messages."""
        mock = MockProvider()
        custom_logger = logging.getLogger("custom_test_logger")
        middleware = LoggingMiddleware(client=mock, logger=custom_logger, log_level="INFO")

        caplog.set_level(logging.INFO, logger="custom_test_logger")
        request = AgentRequest(prompt="test")

        await middleware.send_request(request)

        # Custom logger should have received the log
        assert any(record.name == "custom_test_logger" for record in caplog.records)

    @pytest.mark.asyncio
    async def test_request_id_format_is_uuid(self) -> None:
        """Request IDs are valid UUIDs."""
        import uuid

        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock)

        request_id = middleware._generate_request_id()

        # Should be parseable as UUID
        parsed = uuid.UUID(request_id)
        assert str(parsed) == request_id


class TestResponseLogging:
    """Test response logging with duration and token counts."""

    @pytest.mark.asyncio
    async def test_send_request_logs_response_completion(self, caplog) -> None:
        """send_request logs response completion with duration."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        await middleware.send_request(request)

        # Should have logged both request start and completion
        assert len(caplog.records) == 2
        assert "Request started" in caplog.text
        assert "Request completed" in caplog.text

    @pytest.mark.asyncio
    async def test_response_logging_includes_duration(self, caplog) -> None:
        """Response log includes duration_ms."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        await middleware.send_request(request)

        completion_log = caplog.records[1]
        assert "duration_ms" in completion_log.__dict__
        assert completion_log.__dict__["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_response_logging_includes_model(self, caplog) -> None:
        """Response log includes model name."""
        custom_response = AgentResponse(
            content="test", model="gpt-4", input_tokens=10, output_tokens=20
        )
        mock = MockProvider(responses=[custom_response])
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        await middleware.send_request(request)

        completion_log = caplog.records[1]
        assert completion_log.__dict__["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_response_logging_includes_token_counts(self, caplog) -> None:
        """Response log includes input_tokens, output_tokens, total_tokens."""
        custom_response = AgentResponse(
            content="test", model="test-model", input_tokens=50, output_tokens=100
        )
        mock = MockProvider(responses=[custom_response])
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        await middleware.send_request(request)

        completion_log = caplog.records[1]
        assert completion_log.__dict__["input_tokens"] == 50
        assert completion_log.__dict__["output_tokens"] == 100
        assert completion_log.__dict__["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_response_logging_includes_request_id(self, caplog) -> None:
        """Response log includes same request_id as request log."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        await middleware.send_request(request)

        request_log = caplog.records[0]
        completion_log = caplog.records[1]

        request_id = request_log.__dict__["request_id"]
        assert completion_log.__dict__["request_id"] == request_id

    @pytest.mark.asyncio
    async def test_send_request_stream_logs_response_completion(self, caplog) -> None:
        """send_request_stream logs response completion."""

        class MultiChunkProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="chunk1", is_final=False)
                yield StreamChunk(
                    delta="chunk2",
                    is_final=True,
                    metadata={"model": "test-model", "input_tokens": 10, "output_tokens": 20},
                )

        mock = MultiChunkProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        async for _ in middleware.send_request_stream(request):
            pass

        # Should have logged both request start and completion
        assert len(caplog.records) == 2
        assert "Request started" in caplog.text
        assert "Request completed" in caplog.text

    @pytest.mark.asyncio
    async def test_stream_completion_uses_final_chunk_metadata(self, caplog) -> None:
        """Stream completion logs use metadata from final chunk."""

        class TokenCountingProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="hello", is_final=False)
                yield StreamChunk(
                    delta="world",
                    is_final=True,
                    metadata={"model": "custom-model", "input_tokens": 25, "output_tokens": 50},
                )

        mock = TokenCountingProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        async for _ in middleware.send_request_stream(request):
            pass

        completion_log = caplog.records[1]
        assert completion_log.__dict__["model"] == "custom-model"
        assert completion_log.__dict__["input_tokens"] == 25
        assert completion_log.__dict__["output_tokens"] == 50
        assert completion_log.__dict__["total_tokens"] == 75

    @pytest.mark.asyncio
    async def test_duration_measured_accurately(self, caplog) -> None:
        """Duration is measured from request start to completion."""
        import asyncio

        # Mock with small latency
        mock = MockProvider(latency=0.1)
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        await middleware.send_request(request)

        completion_log = caplog.records[1]
        duration_ms = completion_log.__dict__["duration_ms"]

        # Duration should be at least the latency (100ms)
        assert duration_ms >= 100

    @pytest.mark.asyncio
    async def test_stream_without_final_chunk_no_completion_log(self, caplog) -> None:
        """Stream without final chunk doesn't log completion."""

        class NoFinalChunkProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="chunk1", is_final=False)
                yield StreamChunk(delta="chunk2", is_final=False)

        mock = NoFinalChunkProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        async for _ in middleware.send_request_stream(request):
            pass

        # Should only have request start, no completion
        assert len(caplog.records) == 1
        assert "Request started" in caplog.text


class TestErrorLogging:
    """Test error logging with exceptions and stack traces."""

    @pytest.mark.asyncio
    async def test_send_request_logs_error(self, caplog) -> None:
        """send_request logs errors with exception details."""
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error", status_code=500)

        mock = FailingProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        # Should have logged request start and error
        assert len(caplog.records) == 2
        assert "Request started" in caplog.text
        assert "Request failed" in caplog.text

    @pytest.mark.asyncio
    async def test_error_logging_includes_request_id(self, caplog) -> None:
        """Error log includes same request_id as request log."""
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error")

        mock = FailingProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        request_log = caplog.records[0]
        error_log = caplog.records[1]

        request_id = request_log.__dict__["request_id"]
        assert error_log.__dict__["request_id"] == request_id

    @pytest.mark.asyncio
    async def test_error_logging_includes_exception_type(self, caplog) -> None:
        """Error log includes exception type name."""
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error", status_code=500)

        mock = FailingProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        error_log = caplog.records[1]
        assert error_log.__dict__["error_type"] == "ProviderError"

    @pytest.mark.asyncio
    async def test_error_logging_includes_exception_message(self, caplog) -> None:
        """Error log includes exception message."""
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("Custom error message", status_code=500)

        mock = FailingProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        error_log = caplog.records[1]
        assert error_log.__dict__["error_message"] == "Custom error message"

    @pytest.mark.asyncio
    async def test_error_logging_includes_duration(self, caplog) -> None:
        """Error log includes duration until error occurred."""
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error")

        mock = FailingProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        error_log = caplog.records[1]
        assert "duration_ms" in error_log.__dict__
        assert error_log.__dict__["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_error_logging_includes_stack_trace(self, caplog) -> None:
        """Error log includes stack trace (exc_info=True)."""
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error")

        mock = FailingProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        error_log = caplog.records[1]
        # exc_info should be present in the log record
        assert error_log.exc_info is not None

    @pytest.mark.asyncio
    async def test_exception_propagates_after_logging(self) -> None:
        """Exception is re-raised after logging."""
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error", status_code=500)

        mock = FailingProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        request = AgentRequest(prompt="test")

        # Exception should propagate
        with pytest.raises(ProviderError, match="API error"):
            await middleware.send_request(request)

    @pytest.mark.asyncio
    async def test_send_request_stream_logs_error(self, caplog) -> None:
        """send_request_stream logs errors with exception details."""
        from mada_modelkit._errors import ProviderError

        class FailingStreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                raise ProviderError("Stream error", status_code=500)
                yield  # Make it a generator

        mock = FailingStreamProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            async for _ in middleware.send_request_stream(request):
                pass

        # Should have logged request start and error
        assert len(caplog.records) == 2
        assert "Request started" in caplog.text
        assert "Request failed" in caplog.text

    @pytest.mark.asyncio
    async def test_error_log_level_is_error(self, caplog) -> None:
        """Errors are logged at ERROR level."""
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error")

        mock = FailingProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        error_log = caplog.records[1]
        assert error_log.levelno == logging.ERROR


class TestCorrelationIDPropagation:
    """Test correlation ID generation and propagation."""

    @pytest.mark.asyncio
    async def test_request_with_existing_id_uses_that_id(self, caplog) -> None:
        """Request with existing request_id in metadata uses that ID."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        existing_id = "custom-request-id-123"
        request = AgentRequest(prompt="test", metadata={"request_id": existing_id})

        await middleware.send_request(request)

        # Should use existing ID, not generate new one
        request_log = caplog.records[0]
        assert request_log.__dict__["request_id"] == existing_id

    @pytest.mark.asyncio
    async def test_request_without_id_generates_new_id(self, caplog) -> None:
        """Request without request_id gets new ID generated."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test", metadata={})

        await middleware.send_request(request)

        # Should have generated new ID
        request_log = caplog.records[0]
        assert "request_id" in request_log.__dict__
        assert request_log.__dict__["request_id"] is not None

    @pytest.mark.asyncio
    async def test_request_id_propagated_to_wrapped_client(self) -> None:
        """Request ID is added to metadata when passed to wrapped client."""

        class InspectingProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.received_request = None

            async def send_request(self, request):
                self.received_request = request
                return await super().send_request(request)

        mock = InspectingProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        request = AgentRequest(prompt="test", metadata={})
        await middleware.send_request(request)

        # Wrapped client should receive request with request_id in metadata
        assert mock.received_request is not None
        assert "request_id" in mock.received_request.metadata

    @pytest.mark.asyncio
    async def test_existing_id_preserved_through_propagation(self) -> None:
        """Existing request ID is preserved when propagated."""

        class InspectingProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.received_request = None

            async def send_request(self, request):
                self.received_request = request
                return await super().send_request(request)

        mock = InspectingProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        existing_id = "my-custom-id"
        request = AgentRequest(prompt="test", metadata={"request_id": existing_id})
        await middleware.send_request(request)

        # Wrapped client should receive same ID
        assert mock.received_request.metadata["request_id"] == existing_id

    @pytest.mark.asyncio
    async def test_propagated_request_preserves_all_fields(self) -> None:
        """Propagated request preserves all original fields."""

        class InspectingProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.received_request = None

            async def send_request(self, request):
                self.received_request = request
                return await super().send_request(request)

        mock = InspectingProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        request = AgentRequest(
            prompt="test prompt",
            system_prompt="test system",
            max_tokens=512,
            temperature=0.9,
            stop=["STOP"],
            metadata={"user": "alice"},
        )
        await middleware.send_request(request)

        # All fields should be preserved
        received = mock.received_request
        assert received.prompt == "test prompt"
        assert received.system_prompt == "test system"
        assert received.max_tokens == 512
        assert received.temperature == 0.9
        assert received.stop == ["STOP"]
        assert received.metadata["user"] == "alice"
        assert "request_id" in received.metadata

    @pytest.mark.asyncio
    async def test_send_request_stream_propagates_id(self) -> None:
        """send_request_stream also propagates request ID."""

        class InspectingStreamProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.received_request = None

            async def send_request_stream(self, request):
                self.received_request = request
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="test", is_final=True)

        mock = InspectingStreamProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        existing_id = "stream-id-456"
        request = AgentRequest(prompt="test", metadata={"request_id": existing_id})

        async for _ in middleware.send_request_stream(request):
            pass

        # Wrapped client should receive request with ID
        assert mock.received_request.metadata["request_id"] == existing_id

    @pytest.mark.asyncio
    async def test_multiple_requests_with_same_id_use_that_id(self, caplog) -> None:
        """Multiple requests can share the same request ID."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        shared_id = "shared-id-789"

        # First request
        request1 = AgentRequest(prompt="test1", metadata={"request_id": shared_id})
        await middleware.send_request(request1)

        # Second request with same ID
        request2 = AgentRequest(prompt="test2", metadata={"request_id": shared_id})
        await middleware.send_request(request2)

        # Both should use the shared ID
        assert caplog.records[0].__dict__["request_id"] == shared_id
        assert caplog.records[2].__dict__["request_id"] == shared_id

    @pytest.mark.asyncio
    async def test_get_or_generate_request_id_with_existing_id(self) -> None:
        """_get_or_generate_request_id returns existing ID when present."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock)

        request = AgentRequest(prompt="test", metadata={"request_id": "existing-123"})
        request_id = middleware._get_or_generate_request_id(request)

        assert request_id == "existing-123"

    @pytest.mark.asyncio
    async def test_get_or_generate_request_id_without_existing_id(self) -> None:
        """_get_or_generate_request_id generates new ID when absent."""
        import uuid

        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock)

        request = AgentRequest(prompt="test", metadata={})
        request_id = middleware._get_or_generate_request_id(request)

        # Should be a valid UUID
        parsed = uuid.UUID(request_id)
        assert str(parsed) == request_id


class TestLoggingComprehensive:
    """Comprehensive integration tests for LoggingMiddleware."""

    @pytest.mark.asyncio
    async def test_full_request_lifecycle_logging(self, caplog) -> None:
        """Complete request logs start, completion with all metadata."""
        custom_response = AgentResponse(
            content="response", model="gpt-4", input_tokens=50, output_tokens=100
        )
        mock = MockProvider(responses=[custom_response])
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(
            prompt="test", max_tokens=200, temperature=0.8, metadata={"user": "alice"}
        )

        response = await middleware.send_request(request)

        # Should have 2 logs: request start + completion
        assert len(caplog.records) == 2

        # Request start log
        start_log = caplog.records[0]
        assert start_log.__dict__["event"] == "request_start"
        assert "request_id" in start_log.__dict__
        assert start_log.__dict__["max_tokens"] == 200
        assert start_log.__dict__["temperature"] == 0.8
        assert start_log.__dict__["metadata"]["user"] == "alice"

        # Completion log
        completion_log = caplog.records[1]
        assert completion_log.__dict__["event"] == "request_complete"
        assert completion_log.__dict__["request_id"] == start_log.__dict__["request_id"]
        assert completion_log.__dict__["model"] == "gpt-4"
        assert completion_log.__dict__["input_tokens"] == 50
        assert completion_log.__dict__["output_tokens"] == 100
        assert completion_log.__dict__["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_error_request_lifecycle_logging(self, caplog) -> None:
        """Failed request logs start and error (no completion)."""
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API failed", status_code=500)

        mock = FailingProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        # Should have 2 logs: request start + error (no completion)
        assert len(caplog.records) == 2

        start_log = caplog.records[0]
        assert start_log.__dict__["event"] == "request_start"

        error_log = caplog.records[1]
        assert error_log.__dict__["event"] == "request_error"
        assert error_log.__dict__["request_id"] == start_log.__dict__["request_id"]
        assert error_log.__dict__["error_type"] == "ProviderError"
        assert error_log.__dict__["error_message"] == "API failed"

    @pytest.mark.asyncio
    async def test_pii_filtering_excludes_prompts_by_default(self, caplog) -> None:
        """With include_prompts=False, prompts are not in logs."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO", include_prompts=False)

        caplog.set_level(logging.INFO)
        request = AgentRequest(
            prompt="sensitive user data", system_prompt="system instructions"
        )

        await middleware.send_request(request)

        # Check logs don't contain prompts
        for record in caplog.records:
            assert "prompt" not in record.__dict__
            assert "system_prompt" not in record.__dict__

        # Also check text output
        assert "sensitive user data" not in caplog.text
        assert "system instructions" not in caplog.text

    @pytest.mark.asyncio
    async def test_pii_filtering_includes_prompts_when_enabled(self, caplog) -> None:
        """With include_prompts=True, prompts are in logs."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO", include_prompts=True)

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test prompt", system_prompt="test system")

        await middleware.send_request(request)

        start_log = caplog.records[0]
        assert start_log.__dict__["prompt"] == "test prompt"
        assert start_log.__dict__["system_prompt"] == "test system"

    @pytest.mark.asyncio
    async def test_id_propagation_through_middleware_stack(self) -> None:
        """Request ID propagates through multiple middleware layers."""
        from mada_modelkit.middleware.retry import RetryMiddleware

        class InspectingProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.received_requests = []

            async def send_request(self, request):
                self.received_requests.append(request)
                return await super().send_request(request)

        mock = InspectingProvider()
        retry_mw = RetryMiddleware(client=mock, max_retries=1)
        logging_mw = LoggingMiddleware(client=retry_mw, log_level="INFO")

        custom_id = "trace-id-123"
        request = AgentRequest(prompt="test", metadata={"request_id": custom_id})

        await logging_mw.send_request(request)

        # Provider should receive request with custom ID
        assert len(mock.received_requests) == 1
        assert mock.received_requests[0].metadata["request_id"] == custom_id

    @pytest.mark.asyncio
    async def test_middleware_composition_with_cost_control(self, caplog) -> None:
        """LoggingMiddleware stacks with CostControlMiddleware."""
        from mada_modelkit.middleware.cost_control import CostControlMiddleware

        mock = MockProvider()
        cost_fn = lambda resp: 1.5
        cost_mw = CostControlMiddleware(client=mock, cost_fn=cost_fn)
        logging_mw = LoggingMiddleware(client=cost_mw, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        response = await logging_mw.send_request(request)

        # Both middleware should work
        assert response.content == "mock"
        assert cost_mw.total_spend == 1.5
        assert len(caplog.records) == 2  # Start + completion

    @pytest.mark.asyncio
    async def test_context_manager_support(self, caplog) -> None:
        """LoggingMiddleware works as async context manager."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        async with middleware:
            response = await middleware.send_request(request)

        assert response.content == "mock"
        assert len(caplog.records) == 2

    @pytest.mark.asyncio
    async def test_stream_full_lifecycle_logging(self, caplog) -> None:
        """Stream request logs start and completion."""

        class MultiChunkProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="chunk1", is_final=False)
                yield StreamChunk(delta="chunk2", is_final=False)
                yield StreamChunk(
                    delta="chunk3",
                    is_final=True,
                    metadata={"model": "stream-model", "input_tokens": 15, "output_tokens": 30},
                )

        mock = MultiChunkProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test", metadata={"session": "abc"})

        chunks = []
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        # Should have 2 logs: start + completion
        assert len(caplog.records) == 2

        start_log = caplog.records[0]
        assert start_log.__dict__["event"] == "request_start"
        assert start_log.__dict__["metadata"]["session"] == "abc"

        completion_log = caplog.records[1]
        assert completion_log.__dict__["event"] == "request_complete"
        assert completion_log.__dict__["model"] == "stream-model"
        assert completion_log.__dict__["input_tokens"] == 15
        assert completion_log.__dict__["output_tokens"] == 30

    @pytest.mark.asyncio
    async def test_concurrent_requests_independent_logging(self, caplog) -> None:
        """Concurrent requests have independent request IDs."""
        import asyncio

        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        # Send 3 concurrent requests
        await asyncio.gather(
            middleware.send_request(request),
            middleware.send_request(request),
            middleware.send_request(request),
        )

        # Should have 6 logs (2 per request)
        assert len(caplog.records) == 6

        # Extract request IDs
        request_ids = set()
        for i in range(0, 6, 2):  # Every start log
            request_ids.add(caplog.records[i].__dict__["request_id"])

        # All should be unique
        assert len(request_ids) == 3

    @pytest.mark.asyncio
    async def test_log_output_structured_format(self, caplog) -> None:
        """Log records contain structured data in extra fields."""
        mock = MockProvider()
        middleware = LoggingMiddleware(client=mock, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        await middleware.send_request(request)

        # All logs should have structured data
        for record in caplog.records:
            assert "event" in record.__dict__
            assert "request_id" in record.__dict__
            assert record.__dict__["event"] in ["request_start", "request_complete", "request_error"]

    @pytest.mark.asyncio
    async def test_different_log_levels_filter_correctly(self, caplog) -> None:
        """Different log levels produce expected output."""
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("Error")

        mock = FailingProvider()

        # INFO level: sees request start and errors
        mw_info = LoggingMiddleware(client=mock, log_level="INFO")
        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            await mw_info.send_request(request)

        assert len(caplog.records) == 2  # Start (INFO) + Error (ERROR)

        caplog.clear()

        # ERROR level: only sees errors
        mw_error = LoggingMiddleware(client=mock, log_level="ERROR")
        caplog.set_level(logging.ERROR)

        with pytest.raises(ProviderError):
            await mw_error.send_request(request)

        assert len(caplog.records) == 1  # Only error

    @pytest.mark.asyncio
    async def test_logging_with_timeout_middleware(self, caplog) -> None:
        """LoggingMiddleware works with TimeoutMiddleware."""
        import asyncio
        from mada_modelkit.middleware.timeout import TimeoutMiddleware

        mock = MockProvider()
        timeout_mw = TimeoutMiddleware(client=mock, timeout_seconds=1.0)
        logging_mw = LoggingMiddleware(client=timeout_mw, log_level="INFO")

        caplog.set_level(logging.INFO)
        request = AgentRequest(prompt="test")

        response = await logging_mw.send_request(request)

        assert response.content == "mock"
        assert len(caplog.records) == 2
