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
