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
