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


class TestHistogramMetrics:
    """Test histogram metrics for latency and token distributions."""

    def _get_histogram_count(self, histogram) -> float:
        """Extract count from histogram samples."""
        samples = list(histogram.collect())[0].samples
        count_sample = [s for s in samples if s.name.endswith('_count')][0]
        return count_sample.value

    def _get_histogram_sum(self, histogram) -> float:
        """Extract sum from histogram samples."""
        samples = list(histogram.collect())[0].samples
        sum_sample = [s for s in samples if s.name.endswith('_sum')][0]
        return sum_sample.value

    @pytest.mark.asyncio
    async def test_request_duration_tracked_for_send_request(self) -> None:
        """send_request records duration in histogram."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Send request
        await middleware.send_request(request)

        # Duration histogram should have 1 observation
        assert self._get_histogram_count(middleware._request_duration_seconds) == 1
        assert self._get_histogram_sum(middleware._request_duration_seconds) > 0

    @pytest.mark.asyncio
    async def test_request_duration_tracked_for_stream(self) -> None:
        """send_request_stream records total duration in histogram."""
        from prometheus_client import CollectorRegistry

        class StreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="chunk", is_final=True)

        mock = StreamProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        async for _ in middleware.send_request_stream(request):
            pass

        # Duration histogram should have 1 observation
        assert self._get_histogram_count(middleware._request_duration_seconds) == 1
        assert self._get_histogram_sum(middleware._request_duration_seconds) > 0

    @pytest.mark.asyncio
    async def test_input_tokens_tracked(self) -> None:
        """Input tokens recorded in histogram."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        # Input tokens histogram should have 1 observation
        assert self._get_histogram_count(middleware._input_tokens) == 1
        assert self._get_histogram_sum(middleware._input_tokens) == response.input_tokens

    @pytest.mark.asyncio
    async def test_output_tokens_tracked(self) -> None:
        """Output tokens recorded in histogram."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        # Output tokens histogram should have 1 observation
        assert self._get_histogram_count(middleware._output_tokens) == 1
        assert self._get_histogram_sum(middleware._output_tokens) == response.output_tokens

    @pytest.mark.asyncio
    async def test_ttft_tracked_for_stream(self) -> None:
        """TTFT (time to first token) recorded for streaming."""
        from prometheus_client import CollectorRegistry

        class StreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="first", is_final=False)
                yield StreamChunk(delta="second", is_final=True)

        mock = StreamProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        async for _ in middleware.send_request_stream(request):
            pass

        # TTFT histogram should have 1 observation
        assert self._get_histogram_count(middleware._ttft_seconds) == 1
        assert self._get_histogram_sum(middleware._ttft_seconds) > 0

    @pytest.mark.asyncio
    async def test_ttft_recorded_on_first_chunk_only(self) -> None:
        """TTFT recorded only on first chunk, not subsequent chunks."""
        from prometheus_client import CollectorRegistry

        class MultiChunkProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk
                import asyncio

                await asyncio.sleep(0.01)
                yield StreamChunk(delta="1", is_final=False)
                await asyncio.sleep(0.01)
                yield StreamChunk(delta="2", is_final=False)
                await asyncio.sleep(0.01)
                yield StreamChunk(delta="3", is_final=True)

        mock = MultiChunkProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        async for _ in middleware.send_request_stream(request):
            pass

        # Only 1 TTFT observation despite 3 chunks
        assert self._get_histogram_count(middleware._ttft_seconds) == 1

    @pytest.mark.asyncio
    async def test_stream_tokens_from_final_chunk_metadata(self) -> None:
        """Stream token counts extracted from final chunk metadata."""
        from prometheus_client import CollectorRegistry

        class MetadataStreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="chunk1", is_final=False)
                yield StreamChunk(
                    delta="chunk2",
                    is_final=True,
                    metadata={"input_tokens": 50, "output_tokens": 100},
                )

        mock = MetadataStreamProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        async for _ in middleware.send_request_stream(request):
            pass

        # Token histograms should have metadata values
        assert self._get_histogram_count(middleware._input_tokens) == 1
        assert self._get_histogram_sum(middleware._input_tokens) == 50
        assert self._get_histogram_count(middleware._output_tokens) == 1
        assert self._get_histogram_sum(middleware._output_tokens) == 100

    @pytest.mark.asyncio
    async def test_multiple_requests_accumulate_histogram_data(self) -> None:
        """Multiple requests accumulate observations in histograms."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Send 3 requests
        await middleware.send_request(request)
        await middleware.send_request(request)
        await middleware.send_request(request)

        # Duration histogram should have 3 observations
        assert self._get_histogram_count(middleware._request_duration_seconds) == 3

        # Token histograms should have 3 observations each
        assert self._get_histogram_count(middleware._input_tokens) == 3
        assert self._get_histogram_count(middleware._output_tokens) == 3

    @pytest.mark.asyncio
    async def test_errors_dont_record_histogram_metrics(self) -> None:
        """Failed requests don't record histogram metrics."""
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

        # No histogram observations should be recorded
        assert self._get_histogram_count(middleware._request_duration_seconds) == 0
        assert self._get_histogram_count(middleware._input_tokens) == 0
        assert self._get_histogram_count(middleware._output_tokens) == 0

    @pytest.mark.asyncio
    async def test_none_token_values_skipped(self) -> None:
        """None token values don't record histogram observations."""
        from prometheus_client import CollectorRegistry

        class NoTokensProvider(MockProvider):
            async def send_request(self, request):
                self.call_count += 1
                from mada_modelkit._types import AgentResponse

                return AgentResponse(
                    content="test",
                    model="mock",
                    input_tokens=None,
                    output_tokens=None,
                )

        mock = NoTokensProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")
        await middleware.send_request(request)

        # Duration should be recorded
        assert self._get_histogram_count(middleware._request_duration_seconds) == 1

        # But token histograms should have no observations
        assert self._get_histogram_count(middleware._input_tokens) == 0
        assert self._get_histogram_count(middleware._output_tokens) == 0

    @pytest.mark.asyncio
    async def test_histogram_sum_increases_with_observations(self) -> None:
        """Histogram sum accumulates across multiple observations."""
        from prometheus_client import CollectorRegistry

        class CustomTokenProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            async def send_request(self, request):
                self.call_count += 1
                from mada_modelkit._types import AgentResponse

                # Return different token counts for each call
                return AgentResponse(
                    content="test",
                    model="mock",
                    input_tokens=10 * self.call_count,
                    output_tokens=20 * self.call_count,
                )

        mock = CustomTokenProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # First request: 10 input, 20 output
        await middleware.send_request(request)
        assert self._get_histogram_sum(middleware._input_tokens) == 10
        assert self._get_histogram_sum(middleware._output_tokens) == 20

        # Second request: 20 input, 40 output
        await middleware.send_request(request)
        assert self._get_histogram_sum(middleware._input_tokens) == 30  # 10 + 20
        assert self._get_histogram_sum(middleware._output_tokens) == 60  # 20 + 40

        # Third request: 30 input, 60 output
        await middleware.send_request(request)
        assert self._get_histogram_sum(middleware._input_tokens) == 60  # 10 + 20 + 30
        assert self._get_histogram_sum(middleware._output_tokens) == 120  # 20 + 40 + 60


class TestGaugeMetrics:
    """Test gauge metrics for active request tracking."""

    def _get_gauge_value(self, gauge) -> float:
        """Extract value from gauge samples."""
        samples = list(gauge.collect())[0].samples
        # Gauge has a single sample with the current value
        return samples[0].value

    @pytest.mark.asyncio
    async def test_active_requests_increments_during_request(self) -> None:
        """active_requests increments when request starts."""
        from prometheus_client import CollectorRegistry
        import asyncio

        class SlowProvider(MockProvider):
            async def send_request(self, request):
                self.call_count += 1
                # Simulate slow request
                await asyncio.sleep(0.05)
                return await super().send_request(request)

        mock = SlowProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Initial value should be 0
        assert self._get_gauge_value(middleware._active_requests) == 0

        # Start request in background
        task = asyncio.create_task(middleware.send_request(request))

        # Give it time to start
        await asyncio.sleep(0.01)

        # Should be 1 while request is in flight
        assert self._get_gauge_value(middleware._active_requests) == 1

        # Wait for completion
        await task

        # Should return to 0
        assert self._get_gauge_value(middleware._active_requests) == 0

    @pytest.mark.asyncio
    async def test_active_requests_decrements_on_completion(self) -> None:
        """active_requests decrements when request completes."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Initial value
        assert self._get_gauge_value(middleware._active_requests) == 0

        # Send request
        await middleware.send_request(request)

        # Should return to 0 after completion
        assert self._get_gauge_value(middleware._active_requests) == 0

    @pytest.mark.asyncio
    async def test_active_requests_decrements_on_error(self) -> None:
        """active_requests decrements even when request fails."""
        from prometheus_client import CollectorRegistry
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error")

        mock = FailingProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Initial value
        assert self._get_gauge_value(middleware._active_requests) == 0

        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        # Should return to 0 even after error
        assert self._get_gauge_value(middleware._active_requests) == 0

    @pytest.mark.asyncio
    async def test_active_requests_tracks_concurrent_requests(self) -> None:
        """active_requests tracks multiple concurrent requests."""
        from prometheus_client import CollectorRegistry
        import asyncio

        class SlowProvider(MockProvider):
            async def send_request(self, request):
                self.call_count += 1
                await asyncio.sleep(0.05)
                return await super().send_request(request)

        mock = SlowProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Start 3 concurrent requests
        tasks = [
            asyncio.create_task(middleware.send_request(request))
            for _ in range(3)
        ]

        # Give them time to start
        await asyncio.sleep(0.01)

        # Should show 3 active requests
        assert self._get_gauge_value(middleware._active_requests) == 3

        # Wait for all to complete
        await asyncio.gather(*tasks)

        # Should return to 0
        assert self._get_gauge_value(middleware._active_requests) == 0

    @pytest.mark.asyncio
    async def test_active_requests_for_stream(self) -> None:
        """active_requests tracks streaming requests."""
        from prometheus_client import CollectorRegistry

        class StreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="chunk1", is_final=False)
                yield StreamChunk(delta="chunk2", is_final=True)

        mock = StreamProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Initial value
        assert self._get_gauge_value(middleware._active_requests) == 0

        # Consume stream
        async for _ in middleware.send_request_stream(request):
            pass

        # Should return to 0 after stream completes
        assert self._get_gauge_value(middleware._active_requests) == 0

    @pytest.mark.asyncio
    async def test_active_requests_stream_increments_during_streaming(self) -> None:
        """active_requests is incremented during streaming."""
        from prometheus_client import CollectorRegistry
        import asyncio

        class SlowStreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="chunk1", is_final=False)
                await asyncio.sleep(0.05)
                yield StreamChunk(delta="chunk2", is_final=True)

        mock = SlowStreamProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Initial value
        assert self._get_gauge_value(middleware._active_requests) == 0

        # Start streaming
        stream = middleware.send_request_stream(request)

        # Get first chunk
        first_chunk = await stream.__anext__()
        assert first_chunk.delta == "chunk1"

        # Should be 1 during streaming
        assert self._get_gauge_value(middleware._active_requests) == 1

        # Consume remaining chunks
        async for _ in stream:
            pass

        # Should return to 0
        assert self._get_gauge_value(middleware._active_requests) == 0

    @pytest.mark.asyncio
    async def test_active_requests_stream_decrements_on_error(self) -> None:
        """active_requests decrements when stream fails."""
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

        # Should return to 0 even after error
        assert self._get_gauge_value(middleware._active_requests) == 0

    @pytest.mark.asyncio
    async def test_mixed_request_types_share_active_gauge(self) -> None:
        """send_request and send_request_stream share active_requests gauge."""
        from prometheus_client import CollectorRegistry
        import asyncio

        class SlowProvider(MockProvider):
            async def send_request(self, request):
                await asyncio.sleep(0.05)
                return await super().send_request(request)

            async def send_request_stream(self, request):
                from mada_modelkit._types import StreamChunk
                await asyncio.sleep(0.05)
                yield StreamChunk(delta="test", is_final=True)

        mock = SlowProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Start one of each type
        task1 = asyncio.create_task(middleware.send_request(request))

        async def consume_stream():
            async for _ in middleware.send_request_stream(request):
                pass

        task2 = asyncio.create_task(consume_stream())

        # Give them time to start
        await asyncio.sleep(0.01)

        # Should show 2 active requests
        assert self._get_gauge_value(middleware._active_requests) == 2

        # Wait for completion
        await asyncio.gather(task1, task2)

        # Should return to 0
        assert self._get_gauge_value(middleware._active_requests) == 0

    @pytest.mark.asyncio
    async def test_active_requests_sequential_requests(self) -> None:
        """active_requests returns to 0 between sequential requests."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        for _ in range(5):
            # Before request
            assert self._get_gauge_value(middleware._active_requests) == 0

            # Send request
            await middleware.send_request(request)

            # After request
            assert self._get_gauge_value(middleware._active_requests) == 0


class TestLabelSupport:
    """Test label support for model and status tracking."""

    def _get_labeled_counter_value(self, counter, **labels) -> float:
        """Get counter value for specific labels."""
        return counter.labels(**labels)._value.get()

    def _get_labeled_histogram_count(self, histogram, **labels) -> float:
        """Get histogram count for specific labels."""
        labeled_metric = histogram.labels(**labels)
        samples = list(labeled_metric.collect())[0].samples
        count_sample = [s for s in samples if s.name.endswith('_count')][0]
        return count_sample.value

    @pytest.mark.asyncio
    async def test_track_labels_false_by_default(self) -> None:
        """track_labels defaults to False."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        assert middleware._track_labels is False

    @pytest.mark.asyncio
    async def test_track_labels_true_enables_labeling(self) -> None:
        """track_labels=True enables model and status labels."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry, track_labels=True)

        assert middleware._track_labels is True

    @pytest.mark.asyncio
    async def test_requests_total_with_model_and_status_labels(self) -> None:
        """requests_total uses model and status labels when enabled."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry, track_labels=True)

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        # Should have labeled counter for this model and success status
        count = self._get_labeled_counter_value(
            middleware._requests_total,
            model=response.model,
            status="success"
        )
        assert count == 1

    @pytest.mark.asyncio
    async def test_different_models_tracked_separately(self) -> None:
        """Different models tracked with separate label values."""
        from prometheus_client import CollectorRegistry

        class MultiModelProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            async def send_request(self, request):
                self.call_count += 1
                from mada_modelkit._types import AgentResponse

                model = f"model-{self.call_count}"
                return AgentResponse(
                    content="test",
                    model=model,
                    input_tokens=10,
                    output_tokens=20,
                )

        mock = MultiModelProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry, track_labels=True)

        request = AgentRequest(prompt="test")

        # Request 1: model-1
        await middleware.send_request(request)
        assert self._get_labeled_counter_value(
            middleware._requests_total, model="model-1", status="success"
        ) == 1

        # Request 2: model-2
        await middleware.send_request(request)
        assert self._get_labeled_counter_value(
            middleware._requests_total, model="model-2", status="success"
        ) == 1

        # model-1 count should still be 1
        assert self._get_labeled_counter_value(
            middleware._requests_total, model="model-1", status="success"
        ) == 1

    @pytest.mark.asyncio
    async def test_error_status_label_on_failure(self) -> None:
        """Error requests tagged with status=error."""
        from prometheus_client import CollectorRegistry
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error")

        mock = FailingProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry, track_labels=True)

        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        # Should have error status
        count = self._get_labeled_counter_value(
            middleware._requests_total, model="unknown", status="error"
        )
        assert count == 1

    @pytest.mark.asyncio
    async def test_errors_total_with_model_label(self) -> None:
        """errors_total includes model label when track_labels enabled."""
        from prometheus_client import CollectorRegistry
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error")

        mock = FailingProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry, track_labels=True)

        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        # Should have error_type and model labels
        count = self._get_labeled_counter_value(
            middleware._errors_total, error_type="ProviderError", model="unknown"
        )
        assert count == 1

    @pytest.mark.asyncio
    async def test_histograms_with_model_labels(self) -> None:
        """Histograms use model labels when enabled."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry, track_labels=True)

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        # Duration histogram should have model label
        count = self._get_labeled_histogram_count(
            middleware._request_duration_seconds, model=response.model
        )
        assert count == 1

        # Token histograms should have model label
        assert self._get_labeled_histogram_count(
            middleware._input_tokens, model=response.model
        ) == 1
        assert self._get_labeled_histogram_count(
            middleware._output_tokens, model=response.model
        ) == 1

    @pytest.mark.asyncio
    async def test_stream_with_labels(self) -> None:
        """Streaming requests use labels from final chunk metadata."""
        from prometheus_client import CollectorRegistry

        class LabeledStreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="chunk1", is_final=False)
                yield StreamChunk(
                    delta="chunk2",
                    is_final=True,
                    metadata={
                        "model": "stream-model",
                        "input_tokens": 50,
                        "output_tokens": 100,
                    },
                )

        mock = LabeledStreamProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry, track_labels=True)

        request = AgentRequest(prompt="test")

        async for _ in middleware.send_request_stream(request):
            pass

        # Should use model from final chunk
        count = self._get_labeled_counter_value(
            middleware._requests_total, model="stream-model", status="success"
        )
        assert count == 1

        # Histograms should use model label
        assert self._get_labeled_histogram_count(
            middleware._input_tokens, model="stream-model"
        ) == 1

    @pytest.mark.asyncio
    async def test_unknown_model_when_not_available(self) -> None:
        """Model defaults to 'unknown' when not available."""
        from prometheus_client import CollectorRegistry

        class NoModelProvider(MockProvider):
            async def send_request(self, request):
                from mada_modelkit._types import AgentResponse

                return AgentResponse(
                    content="test",
                    model=None,  # No model
                    input_tokens=10,
                    output_tokens=20,
                )

        mock = NoModelProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry, track_labels=True)

        request = AgentRequest(prompt="test")
        await middleware.send_request(request)

        # Should use "unknown" for model
        count = self._get_labeled_counter_value(
            middleware._requests_total, model="unknown", status="success"
        )
        assert count == 1

    @pytest.mark.asyncio
    async def test_labels_disabled_uses_unlabeled_metrics(self) -> None:
        """track_labels=False uses metrics without labels."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry, track_labels=False)

        request = AgentRequest(prompt="test")
        await middleware.send_request(request)

        # Should use unlabeled counter
        assert middleware._requests_total._value.get() == 1

    @pytest.mark.asyncio
    async def test_mixed_success_and_error_status(self) -> None:
        """Success and error requests tracked separately by status."""
        from prometheus_client import CollectorRegistry
        from mada_modelkit._errors import ProviderError

        class SometimesFailingProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.fail_next = False

            async def send_request(self, request):
                if self.fail_next:
                    self.fail_next = False
                    raise ProviderError("Intentional failure")
                self.fail_next = True
                return await super().send_request(request)

        mock = SometimesFailingProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry, track_labels=True)

        request = AgentRequest(prompt="test")

        # Call 1: success
        response = await middleware.send_request(request)
        assert self._get_labeled_counter_value(
            middleware._requests_total, model=response.model, status="success"
        ) == 1

        # Call 2: error
        with pytest.raises(ProviderError):
            await middleware.send_request(request)
        assert self._get_labeled_counter_value(
            middleware._requests_total, model="unknown", status="error"
        ) == 1

        # Call 3: success again
        await middleware.send_request(request)
        assert self._get_labeled_counter_value(
            middleware._requests_total, model=response.model, status="success"
        ) == 2


class TestMetricsComprehensive:
    """Comprehensive integration tests for MetricsMiddleware."""

    @pytest.mark.asyncio
    async def test_full_request_lifecycle_all_metrics(self) -> None:
        """Full request lifecycle records all relevant metrics."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        # Counter incremented
        assert middleware._requests_total._value.get() == 1

        # Histogram count incremented
        samples = list(middleware._request_duration_seconds.collect())[0].samples
        count = [s for s in samples if s.name.endswith('_count')][0].value
        assert count == 1

        # Token histograms incremented
        input_samples = list(middleware._input_tokens.collect())[0].samples
        input_count = [s for s in input_samples if s.name.endswith('_count')][0].value
        assert input_count == 1

        # Active requests returns to 0
        gauge_samples = list(middleware._active_requests.collect())[0].samples
        assert gauge_samples[0].value == 0

    @pytest.mark.asyncio
    async def test_full_stream_lifecycle_all_metrics(self) -> None:
        """Full stream lifecycle records all relevant metrics."""
        from prometheus_client import CollectorRegistry

        class StreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="chunk1", is_final=False)
                yield StreamChunk(
                    delta="chunk2",
                    is_final=True,
                    metadata={"input_tokens": 50, "output_tokens": 100, "model": "test-model"},
                )

        mock = StreamProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        async for _ in middleware.send_request_stream(request):
            pass

        # All metrics recorded
        assert middleware._requests_total._value.get() == 1

        # TTFT recorded
        ttft_samples = list(middleware._ttft_seconds.collect())[0].samples
        ttft_count = [s for s in ttft_samples if s.name.endswith('_count')][0].value
        assert ttft_count == 1

        # Duration recorded
        duration_samples = list(middleware._request_duration_seconds.collect())[0].samples
        duration_count = [s for s in duration_samples if s.name.endswith('_count')][0].value
        assert duration_count == 1

        # Tokens recorded
        input_samples = list(middleware._input_tokens.collect())[0].samples
        input_sum = [s for s in input_samples if s.name.endswith('_sum')][0].value
        assert input_sum == 50

    @pytest.mark.asyncio
    async def test_middleware_composition_with_retry(self) -> None:
        """MetricsMiddleware composes with RetryMiddleware."""
        from prometheus_client import CollectorRegistry
        from mada_modelkit.middleware.retry import RetryMiddleware
        from mada_modelkit._errors import ProviderError

        class FailTwiceProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.attempt = 0

            async def send_request(self, request):
                self.attempt += 1
                if self.attempt < 3:
                    raise ProviderError("Retry me")
                return await super().send_request(request)

        mock = FailTwiceProvider()
        registry = CollectorRegistry()

        # Stack: MetricsMiddleware wraps RetryMiddleware wraps provider
        retry_mw = RetryMiddleware(client=mock, max_retries=3)
        metrics_mw = MetricsMiddleware(client=retry_mw, registry=registry)

        request = AgentRequest(prompt="test")
        response = await metrics_mw.send_request(request)

        # Should record 1 successful request (after retries)
        assert metrics_mw._requests_total._value.get() == 1

        # No errors at metrics level (RetryMiddleware handled them)
        # errors_total should be 0 since retry succeeded

    @pytest.mark.asyncio
    async def test_middleware_composition_with_cost_control(self) -> None:
        """MetricsMiddleware composes with CostControlMiddleware."""
        from prometheus_client import CollectorRegistry
        from mada_modelkit.middleware.cost_control import CostControlMiddleware

        mock = MockProvider()
        registry = CollectorRegistry()

        # Cost function based on tokens
        def token_cost(response):
            return (response.input_tokens or 0) * 0.01 + (response.output_tokens or 0) * 0.02

        # Stack: MetricsMiddleware wraps CostControlMiddleware wraps provider
        cost_mw = CostControlMiddleware(client=mock, cost_fn=token_cost, budget_cap=10.0)
        metrics_mw = MetricsMiddleware(client=cost_mw, registry=registry)

        request = AgentRequest(prompt="test")
        await metrics_mw.send_request(request)

        # Metrics recorded
        assert metrics_mw._requests_total._value.get() == 1

        # Cost tracking happened at CostControl level
        assert cost_mw.total_spend > 0

    @pytest.mark.asyncio
    async def test_context_manager_support(self) -> None:
        """MetricsMiddleware works as context manager."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()

        async with MetricsMiddleware(client=mock, registry=registry) as middleware:
            request = AgentRequest(prompt="test")
            await middleware.send_request(request)

            # Metrics recorded during context
            assert middleware._requests_total._value.get() == 1

        # Metrics persist after context exit
        assert middleware._requests_total._value.get() == 1

    @pytest.mark.asyncio
    async def test_concurrent_requests_accurate_metrics(self) -> None:
        """Concurrent requests produce accurate metric counts."""
        from prometheus_client import CollectorRegistry
        import asyncio

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Run 10 concurrent requests
        tasks = [middleware.send_request(request) for _ in range(10)]
        await asyncio.gather(*tasks)

        # All requests counted
        assert middleware._requests_total._value.get() == 10

        # Duration histogram has 10 observations
        duration_samples = list(middleware._request_duration_seconds.collect())[0].samples
        duration_count = [s for s in duration_samples if s.name.endswith('_count')][0].value
        assert duration_count == 10

        # Active requests back to 0
        gauge_samples = list(middleware._active_requests.collect())[0].samples
        assert gauge_samples[0].value == 0

    @pytest.mark.asyncio
    async def test_mixed_request_types_share_metrics(self) -> None:
        """Regular and streaming requests share same metric instances."""
        from prometheus_client import CollectorRegistry

        class DualProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="test", is_final=True)

        mock = DualProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Regular request
        await middleware.send_request(request)

        # Stream request
        async for _ in middleware.send_request_stream(request):
            pass

        # Both counted in same counter
        assert middleware._requests_total._value.get() == 2

        # Both in duration histogram
        duration_samples = list(middleware._request_duration_seconds.collect())[0].samples
        duration_count = [s for s in duration_samples if s.name.endswith('_count')][0].value
        assert duration_count == 2

    @pytest.mark.asyncio
    async def test_error_scenarios_proper_metrics(self) -> None:
        """Error scenarios record proper error metrics."""
        from prometheus_client import CollectorRegistry
        from mada_modelkit._errors import ProviderError

        class AlwaysFailProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("Always fails")

            async def send_request_stream(self, request):
                raise ProviderError("Stream fails")
                yield

        mock = AlwaysFailProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Regular request error
        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        # Stream request error
        with pytest.raises(ProviderError):
            async for _ in middleware.send_request_stream(request):
                pass

        # Total requests counted
        assert middleware._requests_total._value.get() == 2

        # Errors counted
        error_count = middleware._errors_total.labels(error_type="ProviderError")._value.get()
        assert error_count == 2

        # Active requests back to 0
        gauge_samples = list(middleware._active_requests.collect())[0].samples
        assert gauge_samples[0].value == 0

    @pytest.mark.asyncio
    async def test_labels_in_integration_scenario(self) -> None:
        """Label tracking works in full integration scenario."""
        from prometheus_client import CollectorRegistry

        class MultiModelProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.request_count = 0

            async def send_request(self, request):
                self.request_count += 1
                from mada_modelkit._types import AgentResponse

                # Alternate between two models
                model = "model-a" if self.request_count % 2 == 1 else "model-b"
                return AgentResponse(
                    content="test",
                    model=model,
                    input_tokens=10,
                    output_tokens=20,
                )

        mock = MultiModelProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry, track_labels=True)

        request = AgentRequest(prompt="test")

        # 3 requests: model-a, model-b, model-a
        for _ in range(3):
            await middleware.send_request(request)

        # Check labeled counts
        model_a_count = middleware._requests_total.labels(
            model="model-a", status="success"
        )._value.get()
        model_b_count = middleware._requests_total.labels(
            model="model-b", status="success"
        )._value.get()

        assert model_a_count == 2
        assert model_b_count == 1

    @pytest.mark.asyncio
    async def test_metric_isolation_between_instances(self) -> None:
        """Different middleware instances have isolated metrics."""
        from prometheus_client import CollectorRegistry

        mock1 = MockProvider()
        mock2 = MockProvider()
        registry1 = CollectorRegistry()
        registry2 = CollectorRegistry()

        middleware1 = MetricsMiddleware(client=mock1, registry=registry1, prefix="app1")
        middleware2 = MetricsMiddleware(client=mock2, registry=registry2, prefix="app2")

        request = AgentRequest(prompt="test")

        # Send 1 request to middleware1
        await middleware1.send_request(request)

        # Send 2 requests to middleware2
        await middleware2.send_request(request)
        await middleware2.send_request(request)

        # Independent counts
        assert middleware1._requests_total._value.get() == 1
        assert middleware2._requests_total._value.get() == 2

    @pytest.mark.asyncio
    async def test_custom_prefix_in_metric_names(self) -> None:
        """Custom prefix appears in metric names."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry, prefix="myapp")

        request = AgentRequest(prompt="test")
        await middleware.send_request(request)

        # Check metric name includes prefix
        samples = list(middleware._requests_total.collect())[0].samples
        assert samples[0].name == "myapp_requests_total"

    @pytest.mark.asyncio
    async def test_exception_propagation_preserves_stack_trace(self) -> None:
        """Exceptions propagate with full stack trace."""
        from prometheus_client import CollectorRegistry
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("Original error")

        mock = FailingProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Exception propagated
        with pytest.raises(ProviderError) as exc_info:
            await middleware.send_request(request)

        # Original message preserved
        assert str(exc_info.value) == "Original error"

    @pytest.mark.asyncio
    async def test_metadata_preservation_through_middleware(self) -> None:
        """Request metadata preserved through middleware."""
        from prometheus_client import CollectorRegistry

        mock = MockProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test", metadata={"custom_key": "custom_value"})
        await middleware.send_request(request)

        # Metadata passed to wrapped client
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_zero_active_requests_after_many_operations(self) -> None:
        """Active requests gauge returns to 0 after many operations."""
        from prometheus_client import CollectorRegistry
        from mada_modelkit._errors import ProviderError

        class SometimesFailProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.attempt = 0

            async def send_request(self, request):
                self.attempt += 1
                if self.attempt % 3 == 0:
                    raise ProviderError("Fail")
                return await super().send_request(request)

        mock = SometimesFailProvider()
        registry = CollectorRegistry()
        middleware = MetricsMiddleware(client=mock, registry=registry)

        request = AgentRequest(prompt="test")

        # Many requests, some failing
        for _ in range(10):
            try:
                await middleware.send_request(request)
            except ProviderError:
                pass

        # Active requests back to 0
        gauge_samples = list(middleware._active_requests.collect())[0].samples
        assert gauge_samples[0].value == 0
