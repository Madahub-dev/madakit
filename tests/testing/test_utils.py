"""Tests for testing utilities.

Covers MockProvider enhancements and assertion helpers.
"""

from __future__ import annotations

import time

import pytest

from madakit._types import AgentRequest, AgentResponse
from madakit.testing.utils import (
    MockProvider,
    assert_cache_hit,
    assert_cache_miss,
    assert_retry_count,
    assert_response_time,
)


class TestMockProviderEnhancements:
    """Test enhanced MockProvider."""

    @pytest.mark.asyncio
    async def test_configurable_latency(self) -> None:
        """MockProvider supports configurable latency."""
        provider = MockProvider(latency=0.1)
        request = AgentRequest(prompt="test")

        start = time.perf_counter()
        await provider.send_request(request)
        duration = time.perf_counter() - start

        assert duration >= 0.1

    @pytest.mark.asyncio
    async def test_error_injection(self) -> None:
        """MockProvider supports error injection."""
        provider = MockProvider(errors=[ValueError("test error")])
        request = AgentRequest(prompt="test")

        with pytest.raises(ValueError, match="test error"):
            await provider.send_request(request)

    @pytest.mark.asyncio
    async def test_fail_on_request(self) -> None:
        """MockProvider can be configured to always fail."""
        provider = MockProvider(fail_on_request=True)
        request = AgentRequest(prompt="test")

        with pytest.raises(RuntimeError, match="Mock provider error"):
            await provider.send_request(request)

    @pytest.mark.asyncio
    async def test_call_tracking(self) -> None:
        """MockProvider tracks call count."""
        provider = MockProvider()
        request = AgentRequest(prompt="test")

        assert provider.call_count == 0

        await provider.send_request(request)
        assert provider.call_count == 1

        await provider.send_request(request)
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_last_request_tracking(self) -> None:
        """MockProvider tracks last request."""
        provider = MockProvider()
        request = AgentRequest(prompt="test prompt")

        await provider.send_request(request)

        assert provider.last_request is not None
        assert provider.last_request.prompt == "test prompt"

    @pytest.mark.asyncio
    async def test_stream_count_tracking(self) -> None:
        """MockProvider tracks stream count."""
        provider = MockProvider()
        request = AgentRequest(prompt="test")

        assert provider.stream_count == 0

        async for _ in provider.send_request_stream(request):
            pass

        assert provider.stream_count == 1

    @pytest.mark.asyncio
    async def test_custom_stream_chunks(self) -> None:
        """MockProvider supports custom stream chunks."""
        provider = MockProvider(stream_chunks=["A", "B", "C"])
        request = AgentRequest(prompt="test")

        chunks = []
        async for chunk in provider.send_request_stream(request):
            chunks.append(chunk.delta)

        assert chunks == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_fail_on_stream(self) -> None:
        """MockProvider can fail on streaming."""
        provider = MockProvider(fail_on_stream=True)
        request = AgentRequest(prompt="test")

        with pytest.raises(RuntimeError, match="Mock stream error"):
            async for _ in provider.send_request_stream(request):
                pass


class TestAssertCacheHit:
    """Test assert_cache_hit helper."""

    def test_assert_cache_hit_passes(self) -> None:
        """assert_cache_hit passes when cache_hit is True."""
        response = AgentResponse(
            content="test",
            model="test",
            input_tokens=1,
            output_tokens=1,
            metadata={"cache_hit": True},
        )

        assert_cache_hit(response)

    def test_assert_cache_hit_fails_on_miss(self) -> None:
        """assert_cache_hit fails when cache_hit is False."""
        response = AgentResponse(
            content="test",
            model="test",
            input_tokens=1,
            output_tokens=1,
            metadata={"cache_hit": False},
        )

        with pytest.raises(AssertionError, match="Expected cache hit"):
            assert_cache_hit(response)

    def test_assert_cache_hit_fails_no_metadata(self) -> None:
        """assert_cache_hit fails when no cache_hit in metadata."""
        response = AgentResponse(
            content="test",
            model="test",
            input_tokens=1,
            output_tokens=1,
        )

        with pytest.raises(AssertionError, match="Expected cache hit"):
            assert_cache_hit(response)


class TestAssertCacheMiss:
    """Test assert_cache_miss helper."""

    def test_assert_cache_miss_passes(self) -> None:
        """assert_cache_miss passes when cache_hit is False."""
        response = AgentResponse(
            content="test",
            model="test",
            input_tokens=1,
            output_tokens=1,
            metadata={"cache_hit": False},
        )

        assert_cache_miss(response)

    def test_assert_cache_miss_passes_no_metadata(self) -> None:
        """assert_cache_miss passes when no metadata."""
        response = AgentResponse(
            content="test",
            model="test",
            input_tokens=1,
            output_tokens=1,
        )

        assert_cache_miss(response)

    def test_assert_cache_miss_fails_on_hit(self) -> None:
        """assert_cache_miss fails when cache_hit is True."""
        response = AgentResponse(
            content="test",
            model="test",
            input_tokens=1,
            output_tokens=1,
            metadata={"cache_hit": True},
        )

        with pytest.raises(AssertionError, match="Expected cache miss"):
            assert_cache_miss(response)


class TestAssertRetryCount:
    """Test assert_retry_count helper."""

    def test_assert_retry_count_passes(self) -> None:
        """assert_retry_count passes when count matches."""
        response = AgentResponse(
            content="test",
            model="test",
            input_tokens=1,
            output_tokens=1,
            metadata={"retry_count": 2},
        )

        assert_retry_count(response, 2)

    def test_assert_retry_count_fails_mismatch(self) -> None:
        """assert_retry_count fails when count doesn't match."""
        response = AgentResponse(
            content="test",
            model="test",
            input_tokens=1,
            output_tokens=1,
            metadata={"retry_count": 1},
        )

        with pytest.raises(AssertionError, match="Expected 2 retries"):
            assert_retry_count(response, 2)

    def test_assert_retry_count_defaults_to_zero(self) -> None:
        """assert_retry_count defaults to 0 when no retry_count."""
        response = AgentResponse(
            content="test",
            model="test",
            input_tokens=1,
            output_tokens=1,
        )

        # Passes because metadata defaults and retry_count defaults to 0
        assert_retry_count(response, 0)


class TestAssertResponseTime:
    """Test assert_response_time helper."""

    def test_assert_response_time_passes(self) -> None:
        """assert_response_time passes when within limit."""
        assert_response_time(0.5, 1.0)

    def test_assert_response_time_fails(self) -> None:
        """assert_response_time fails when exceeds limit."""
        with pytest.raises(AssertionError, match="exceeding limit"):
            assert_response_time(2.0, 1.0)

    def test_assert_response_time_exact_limit(self) -> None:
        """assert_response_time passes at exact limit."""
        assert_response_time(1.0, 1.0)
