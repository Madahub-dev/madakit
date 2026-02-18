"""Tests for RetryMiddleware constructor, _default_is_retryable, and send_request
(tasks 2.1.1–2.1.3).

Covers: attribute storage, default values, custom is_retryable predicate,
BaseAgentClient inheritance, semaphore passthrough via super().__init__(),
_default_is_retryable classification of ProviderError status codes and
non-ProviderError exceptions, retry loop correctness, exponential backoff
sleep timing, and RetryExhaustedError on exhaustion.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from helpers import MockProvider
from mada_modelkit._base import BaseAgentClient
from mada_modelkit._errors import ProviderError, RetryExhaustedError
from mada_modelkit.middleware.retry import RetryMiddleware


class TestRetryMiddlewareConstructor:
    """RetryMiddleware.__init__ — attribute storage and defaults."""

    def test_stores_client(self) -> None:
        """Asserts that the wrapped client is stored as _client."""
        provider = MockProvider()
        middleware = RetryMiddleware(client=provider)
        assert middleware._client is provider

    def test_default_max_retries(self) -> None:
        """Asserts that max_retries defaults to 3."""
        middleware = RetryMiddleware(client=MockProvider())
        assert middleware._max_retries == 3

    def test_custom_max_retries(self) -> None:
        """Asserts that a custom max_retries value is stored correctly."""
        middleware = RetryMiddleware(client=MockProvider(), max_retries=5)
        assert middleware._max_retries == 5

    def test_default_backoff_base(self) -> None:
        """Asserts that backoff_base defaults to 1.0."""
        middleware = RetryMiddleware(client=MockProvider())
        assert middleware._backoff_base == 1.0

    def test_custom_backoff_base(self) -> None:
        """Asserts that a custom backoff_base value is stored correctly."""
        middleware = RetryMiddleware(client=MockProvider(), backoff_base=0.5)
        assert middleware._backoff_base == 0.5

    def test_default_is_retryable_is_none(self) -> None:
        """Asserts that is_retryable defaults to None."""
        middleware = RetryMiddleware(client=MockProvider())
        assert middleware._is_retryable is None

    def test_custom_is_retryable_stored(self) -> None:
        """Asserts that a custom is_retryable callable is stored correctly."""
        predicate = lambda exc: isinstance(exc, ValueError)
        middleware = RetryMiddleware(client=MockProvider(), is_retryable=predicate)
        assert middleware._is_retryable is predicate

    def test_is_base_agent_client(self) -> None:
        """Asserts that RetryMiddleware is a BaseAgentClient subclass."""
        middleware = RetryMiddleware(client=MockProvider())
        assert isinstance(middleware, BaseAgentClient)

    def test_no_semaphore_by_default(self) -> None:
        """Asserts that _semaphore is None when max_concurrent is not passed."""
        middleware = RetryMiddleware(client=MockProvider())
        assert middleware._semaphore is None

    def test_zero_max_retries_stored(self) -> None:
        """Asserts that max_retries=0 (no retries) is stored correctly."""
        middleware = RetryMiddleware(client=MockProvider(), max_retries=0)
        assert middleware._max_retries == 0

    def test_float_backoff_base_stored(self) -> None:
        """Asserts that a float backoff_base (e.g. 2.5) is stored as-is."""
        middleware = RetryMiddleware(client=MockProvider(), backoff_base=2.5)
        assert middleware._backoff_base == 2.5

    def test_client_is_itself_a_middleware(self) -> None:
        """Asserts that a middleware can wrap another middleware (composition)."""
        inner = MockProvider()
        outer_retry = RetryMiddleware(client=inner)
        double_retry = RetryMiddleware(client=outer_retry)
        assert double_retry._client is outer_retry
        assert outer_retry._client is inner


class TestRetryMiddlewareStubDelegation:
    """RetryMiddleware stub send_request — delegates to wrapped client."""

    @pytest.mark.asyncio
    async def test_send_request_delegates_to_client(self) -> None:
        """Asserts that send_request passes through to the wrapped client."""
        from mada_modelkit._types import AgentRequest, AgentResponse

        expected = AgentResponse(
            content="hello", model="mock", input_tokens=5, output_tokens=3
        )
        provider = MockProvider(responses=[expected])
        middleware = RetryMiddleware(client=provider)

        request = AgentRequest(prompt="hi")
        result = await middleware.send_request(request)
        assert result is expected
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_send_request_stream_delegates_to_client(self) -> None:
        """Asserts that send_request_stream passes chunks through from wrapped client."""
        from mada_modelkit._types import AgentRequest

        provider = MockProvider()
        middleware = RetryMiddleware(client=provider)

        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="hi")):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].delta == "mock"
        assert chunks[0].is_final is True


class TestDefaultIsRetryable:
    """RetryMiddleware._default_is_retryable — retryability classification."""

    def test_provider_error_none_status_is_retryable(self) -> None:
        """Asserts that ProviderError with status_code=None is retryable."""
        assert RetryMiddleware._default_is_retryable(ProviderError("oops")) is True

    def test_provider_error_429_is_retryable(self) -> None:
        """Asserts that ProviderError with status_code=429 (rate limited) is retryable."""
        assert RetryMiddleware._default_is_retryable(ProviderError("rate limited", 429)) is True

    def test_provider_error_500_is_retryable(self) -> None:
        """Asserts that ProviderError with status_code=500 is retryable."""
        assert RetryMiddleware._default_is_retryable(ProviderError("server error", 500)) is True

    def test_provider_error_503_is_retryable(self) -> None:
        """Asserts that ProviderError with status_code=503 is retryable."""
        assert RetryMiddleware._default_is_retryable(ProviderError("unavailable", 503)) is True

    def test_provider_error_599_is_retryable(self) -> None:
        """Asserts that ProviderError with status_code=599 (any >=500) is retryable."""
        assert RetryMiddleware._default_is_retryable(ProviderError("error", 599)) is True

    def test_provider_error_400_not_retryable(self) -> None:
        """Asserts that ProviderError with status_code=400 (bad request) is not retryable."""
        assert RetryMiddleware._default_is_retryable(ProviderError("bad request", 400)) is False

    def test_provider_error_401_not_retryable(self) -> None:
        """Asserts that ProviderError with status_code=401 (unauthorized) is not retryable."""
        assert RetryMiddleware._default_is_retryable(ProviderError("unauthorized", 401)) is False

    def test_provider_error_403_not_retryable(self) -> None:
        """Asserts that ProviderError with status_code=403 (forbidden) is not retryable."""
        assert RetryMiddleware._default_is_retryable(ProviderError("forbidden", 403)) is False

    def test_provider_error_404_not_retryable(self) -> None:
        """Asserts that ProviderError with status_code=404 (not found) is not retryable."""
        assert RetryMiddleware._default_is_retryable(ProviderError("not found", 404)) is False

    def test_provider_error_422_not_retryable(self) -> None:
        """Asserts that ProviderError with status_code=422 (unprocessable) is not retryable."""
        assert RetryMiddleware._default_is_retryable(ProviderError("unprocessable", 422)) is False

    def test_value_error_not_retryable(self) -> None:
        """Asserts that a plain ValueError is not retryable."""
        assert RetryMiddleware._default_is_retryable(ValueError("bad value")) is False

    def test_os_error_not_retryable(self) -> None:
        """Asserts that an OSError is not retryable."""
        assert RetryMiddleware._default_is_retryable(OSError("io error")) is False

    def test_runtime_error_not_retryable(self) -> None:
        """Asserts that a RuntimeError is not retryable."""
        assert RetryMiddleware._default_is_retryable(RuntimeError("runtime")) is False

    def test_callable_via_instance(self) -> None:
        """Asserts that _default_is_retryable is accessible via an instance."""
        middleware = RetryMiddleware(client=MockProvider())
        assert middleware._default_is_retryable(ProviderError("err", 500)) is True


class TestSendRequest:
    """RetryMiddleware.send_request — retry loop, backoff, and exhaustion."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        """Asserts that a successful first attempt returns without any retry."""
        from mada_modelkit._types import AgentRequest, AgentResponse

        expected = AgentResponse(content="ok", model="m", input_tokens=1, output_tokens=1)
        provider = MockProvider(responses=[expected])
        middleware = RetryMiddleware(client=provider, max_retries=3)

        result = await middleware.send_request(AgentRequest(prompt="hi"))
        assert result is expected
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_retryable_error_retried_until_success(self) -> None:
        """Asserts that a retryable error is retried and returns once provider succeeds."""
        from mada_modelkit._types import AgentRequest, AgentResponse

        success = AgentResponse(content="ok", model="m", input_tokens=1, output_tokens=1)
        provider = MockProvider(
            errors=[ProviderError("server error", 500), ProviderError("server error", 500)],
            responses=[success],
        )
        with patch("mada_modelkit.middleware.retry.asyncio.sleep", new_callable=AsyncMock):
            middleware = RetryMiddleware(client=provider, max_retries=3)
            result = await middleware.send_request(AgentRequest(prompt="hi"))

        assert result is success
        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error_raised_immediately(self) -> None:
        """Asserts that a non-retryable error is re-raised on the first attempt."""
        from mada_modelkit._types import AgentRequest

        err = ProviderError("bad request", 400)
        provider = MockProvider(errors=[err])
        middleware = RetryMiddleware(client=provider, max_retries=3)

        with pytest.raises(ProviderError) as exc_info:
            await middleware.send_request(AgentRequest(prompt="hi"))

        assert exc_info.value is err
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_non_retryable_not_wrapped_in_retry_exhausted(self) -> None:
        """Asserts that non-retryable errors are NOT wrapped in RetryExhaustedError."""
        from mada_modelkit._types import AgentRequest

        provider = MockProvider(errors=[ProviderError("forbidden", 403)])
        middleware = RetryMiddleware(client=provider, max_retries=3)

        with pytest.raises(ProviderError):
            await middleware.send_request(AgentRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_exhaustion_raises_retry_exhausted_error(self) -> None:
        """Asserts that RetryExhaustedError is raised when all retries are consumed."""
        from mada_modelkit._types import AgentRequest

        provider = MockProvider(errors=[ProviderError("err", 500)] * 4)
        with patch("mada_modelkit.middleware.retry.asyncio.sleep", new_callable=AsyncMock):
            middleware = RetryMiddleware(client=provider, max_retries=3)
            with pytest.raises(RetryExhaustedError):
                await middleware.send_request(AgentRequest(prompt="hi"))

        assert provider.call_count == 4  # 1 initial + 3 retries

    @pytest.mark.asyncio
    async def test_retry_exhausted_stores_last_error(self) -> None:
        """Asserts that RetryExhaustedError.last_error is the final caught exception."""
        from mada_modelkit._types import AgentRequest

        last = ProviderError("last error", 503)
        provider = MockProvider(
            errors=[ProviderError("first", 500), ProviderError("second", 502), last]
        )
        with patch("mada_modelkit.middleware.retry.asyncio.sleep", new_callable=AsyncMock):
            middleware = RetryMiddleware(client=provider, max_retries=2)
            with pytest.raises(RetryExhaustedError) as exc_info:
                await middleware.send_request(AgentRequest(prompt="hi"))

        assert exc_info.value.last_error is last

    @pytest.mark.asyncio
    async def test_backoff_sleep_called_with_correct_delays(self) -> None:
        """Asserts that asyncio.sleep is called with backoff_base * 2**attempt."""
        from mada_modelkit._types import AgentRequest

        provider = MockProvider(errors=[ProviderError("err", 500)] * 4)
        mock_sleep = AsyncMock()
        with patch("mada_modelkit.middleware.retry.asyncio.sleep", mock_sleep):
            middleware = RetryMiddleware(client=provider, max_retries=3, backoff_base=1.0)
            with pytest.raises(RetryExhaustedError):
                await middleware.send_request(AgentRequest(prompt="hi"))

        assert mock_sleep.call_count == 3
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls == [1.0, 2.0, 4.0]  # 1.0*2^0, 1.0*2^1, 1.0*2^2

    @pytest.mark.asyncio
    async def test_backoff_uses_backoff_base(self) -> None:
        """Asserts that the backoff_base multiplier scales sleep durations correctly."""
        from mada_modelkit._types import AgentRequest

        provider = MockProvider(errors=[ProviderError("err", 500)] * 3)
        mock_sleep = AsyncMock()
        with patch("mada_modelkit.middleware.retry.asyncio.sleep", mock_sleep):
            middleware = RetryMiddleware(client=provider, max_retries=2, backoff_base=0.5)
            with pytest.raises(RetryExhaustedError):
                await middleware.send_request(AgentRequest(prompt="hi"))

        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls == [0.5, 1.0]  # 0.5*2^0, 0.5*2^1

    @pytest.mark.asyncio
    async def test_max_retries_zero_means_single_attempt(self) -> None:
        """Asserts that max_retries=0 results in exactly one attempt with no sleep."""
        from mada_modelkit._types import AgentRequest

        provider = MockProvider(errors=[ProviderError("err", 500)])
        mock_sleep = AsyncMock()
        with patch("mada_modelkit.middleware.retry.asyncio.sleep", mock_sleep):
            middleware = RetryMiddleware(client=provider, max_retries=0)
            with pytest.raises(RetryExhaustedError):
                await middleware.send_request(AgentRequest(prompt="hi"))

        assert provider.call_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_is_retryable_overrides_default(self) -> None:
        """Asserts that a custom is_retryable predicate controls retry decisions."""
        from mada_modelkit._types import AgentRequest, AgentResponse

        # Custom predicate: retry only on ValueError (not the default ProviderError logic)
        success = AgentResponse(content="ok", model="m", input_tokens=1, output_tokens=1)
        provider = MockProvider(errors=[ValueError("transient")], responses=[success])

        with patch("mada_modelkit.middleware.retry.asyncio.sleep", new_callable=AsyncMock):
            middleware = RetryMiddleware(
                client=provider,
                max_retries=1,
                is_retryable=lambda exc: isinstance(exc, ValueError),
            )
            result = await middleware.send_request(AgentRequest(prompt="hi"))

        assert result is success
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_no_sleep_after_final_attempt(self) -> None:
        """Asserts that asyncio.sleep is not called after the last failed attempt."""
        from mada_modelkit._types import AgentRequest

        # max_retries=2 → 3 total attempts → 2 sleeps (after attempt 0 and 1, not after 2)
        provider = MockProvider(errors=[ProviderError("err", 500)] * 3)
        mock_sleep = AsyncMock()
        with patch("mada_modelkit.middleware.retry.asyncio.sleep", mock_sleep):
            middleware = RetryMiddleware(client=provider, max_retries=2)
            with pytest.raises(RetryExhaustedError):
                await middleware.send_request(AgentRequest(prompt="hi"))

        assert mock_sleep.call_count == 2  # only between attempts, not after last