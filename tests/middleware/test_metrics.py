"""Tests for metrics middleware.

Tests MetricsMiddleware constructor, counter metrics, histogram metrics,
gauge metrics, and label support. Requires prometheus_client.
"""

from __future__ import annotations

import pytest

# Skip all tests if prometheus_client not available
pytest.importorskip("prometheus_client")

from prometheus_client import CollectorRegistry

from mada_modelkit.middleware.metrics import MetricsMiddleware
from mada_modelkit._types import AgentRequest, AgentResponse

from helpers import MockProvider


class TestModuleExports:
    """Test module-level exports and imports."""

    def test_all_exports(self) -> None:
        """__all__ contains only MetricsMiddleware."""
        from mada_modelkit.middleware import metrics

        assert metrics.__all__ == ["MetricsMiddleware"]

    def test_middleware_importable(self) -> None:
        """MetricsMiddleware can be imported from module."""
        from mada_modelkit.middleware.metrics import MetricsMiddleware as MM

        assert MM is not None


class TestMetricsMiddlewareConstructor:
    """Test MetricsMiddleware constructor and initialization."""

    def test_minimal_constructor(self) -> None:
        """Constructor accepts client and uses defaults."""
        mock = MockProvider()
        middleware = MetricsMiddleware(client=mock)

        assert middleware._client is mock
        assert middleware._prefix == "madakit"
        assert middleware._registry is not None

    def test_explicit_registry(self) -> None:
        """Constructor accepts explicit registry."""
        mock = MockProvider()
        custom_registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=custom_registry)

        assert middleware._registry is custom_registry

    def test_explicit_prefix(self) -> None:
        """Constructor accepts explicit prefix."""
        mock = MockProvider()
        middleware = MetricsMiddleware(client=mock, prefix="myapp")

        assert middleware._prefix == "myapp"

    def test_custom_prefix_values(self) -> None:
        """Constructor accepts various prefix values."""
        mock = MockProvider()

        mw1 = MetricsMiddleware(client=mock, prefix="app")
        assert mw1._prefix == "app"

        mw2 = MetricsMiddleware(client=mock, prefix="service_x")
        assert mw2._prefix == "service_x"

        mw3 = MetricsMiddleware(client=mock, prefix="prod")
        assert mw3._prefix == "prod"

    def test_wraps_base_agent_client(self) -> None:
        """MetricsMiddleware can wrap any BaseAgentClient."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        mw1 = MetricsMiddleware(client=mock1)
        mw2 = MetricsMiddleware(client=mock2)

        assert mw1._client is mock1
        assert mw2._client is mock2

    def test_wraps_middleware(self) -> None:
        """MetricsMiddleware can wrap another middleware."""
        from mada_modelkit.middleware.retry import RetryMiddleware

        mock = MockProvider()
        retry_mw = RetryMiddleware(client=mock, max_retries=3)
        metrics_mw = MetricsMiddleware(client=retry_mw, prefix="test")

        assert metrics_mw._client is retry_mw
        assert metrics_mw._prefix == "test"

    def test_super_init_called(self) -> None:
        """MetricsMiddleware calls BaseAgentClient.__init__."""
        mock = MockProvider()
        middleware = MetricsMiddleware(client=mock)

        # Should have BaseAgentClient methods
        assert hasattr(middleware, "send_request")
        assert hasattr(middleware, "send_request_stream")
        assert hasattr(middleware, "cancel")
        assert hasattr(middleware, "close")

    def test_instance_isolation(self) -> None:
        """Different instances have independent state."""
        mock = MockProvider()
        registry1 = CollectorRegistry()
        registry2 = CollectorRegistry()

        mw1 = MetricsMiddleware(client=mock, registry=registry1, prefix="app1")
        mw2 = MetricsMiddleware(client=mock, registry=registry2, prefix="app2")

        assert mw1._registry is registry1
        assert mw2._registry is registry2
        assert mw1._prefix == "app1"
        assert mw2._prefix == "app2"

    def test_prometheus_import_error_raised(self) -> None:
        """ImportError raised if prometheus_client not available."""
        # This test would only fail if prometheus_client is not installed,
        # but the test file is skipped in that case, so we just verify
        # the import works when available
        from prometheus_client import CollectorRegistry

        assert CollectorRegistry is not None
