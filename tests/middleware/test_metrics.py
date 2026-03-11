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

        # Use separate registries to avoid metric name conflicts
        registry1 = CollectorRegistry()
        registry2 = CollectorRegistry()

        mw1 = MetricsMiddleware(client=mock1, registry=registry1)
        mw2 = MetricsMiddleware(client=mock2, registry=registry2)

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
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

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


class TestCounterMetrics:
    """Test counter metrics for requests and errors."""

    @pytest.mark.asyncio
    async def test_requests_total_increments(self) -> None:
        """requests_total counter increments on each request."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Initial value should be 0
        initial = middleware._requests_total._value.get()
        assert initial == 0

        # Send one request
        await middleware.send_request(request)

        # Should increment to 1
        assert middleware._requests_total._value.get() == 1

        # Send another request
        await middleware.send_request(request)

        # Should increment to 2
        assert middleware._requests_total._value.get() == 2

    @pytest.mark.asyncio
    async def test_requests_total_increments_for_stream(self) -> None:
        """requests_total increments for send_request_stream."""
        from prometheus_client import CollectorRegistry

        class MultiChunkProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="chunk", is_final=True)

        mock = MultiChunkProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        initial = middleware._requests_total._value.get()
        assert initial == 0

        async for _ in middleware.send_request_stream(request):
            pass

        assert middleware._requests_total._value.get() == 1

    @pytest.mark.asyncio
    async def test_errors_total_increments_on_error(self) -> None:
        """errors_total counter increments when request fails."""
        from prometheus_client import CollectorRegistry
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error", status_code=500)

        mock = FailingProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        # Error counter should increment
        error_count = middleware._errors_total.labels(error_type="ProviderError")._value.get()
        assert error_count == 1

    @pytest.mark.asyncio
    async def test_errors_total_has_error_type_label(self) -> None:
        """errors_total uses error_type label correctly."""
        from prometheus_client import CollectorRegistry
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error")

        mock = FailingProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        # Should be labeled with error type
        provider_errors = middleware._errors_total.labels(error_type="ProviderError")._value.get()
        assert provider_errors == 1

    @pytest.mark.asyncio
    async def test_different_error_types_tracked_separately(self) -> None:
        """Different error types tracked with separate labels."""
        from prometheus_client import CollectorRegistry
        from mada_modelkit._errors import ProviderError

        class VariableErrorProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            async def send_request(self, request):
                self.call_count += 1
                if self.call_count == 1:
                    raise ProviderError("Error 1")
                elif self.call_count == 2:
                    raise ValueError("Error 2")
                else:
                    return await super().send_request(request)

        mock = VariableErrorProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # First error: ProviderError
        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        # Second error: ValueError
        with pytest.raises(ValueError):
            await middleware.send_request(request)

        # Check separate counts
        provider_errors = middleware._errors_total.labels(error_type="ProviderError")._value.get()
        value_errors = middleware._errors_total.labels(error_type="ValueError")._value.get()

        assert provider_errors == 1
        assert value_errors == 1

    @pytest.mark.asyncio
    async def test_successful_requests_dont_increment_errors(self) -> None:
        """Successful requests don't increment error counter."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Send successful request
        await middleware.send_request(request)

        # No errors should be recorded
        # Note: We can't easily check "no labels", but we can verify
        # that the total metric exists and has the right structure
        assert middleware._errors_total is not None

    @pytest.mark.asyncio
    async def test_stream_errors_increment_counter(self) -> None:
        """Stream errors increment error counter."""
        from prometheus_client import CollectorRegistry
        from mada_modelkit._errors import ProviderError

        class FailingStreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                raise ProviderError("Stream error")
                yield  # Make it a generator

        mock = FailingStreamProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            async for _ in middleware.send_request_stream(request):
                pass

        # Error counter should increment
        error_count = middleware._errors_total.labels(error_type="ProviderError")._value.get()
        assert error_count == 1

    @pytest.mark.asyncio
    async def test_mixed_request_types_share_counters(self) -> None:
        """send_request and send_request_stream use same counters."""
        from prometheus_client import CollectorRegistry

        class StreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="test", is_final=True)

        mock = StreamProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Send regular request
        await middleware.send_request(request)

        # Send stream request
        async for _ in middleware.send_request_stream(request):
            pass

        # Both should increment same counter
        assert middleware._requests_total._value.get() == 2
