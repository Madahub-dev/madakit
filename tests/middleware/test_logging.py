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
