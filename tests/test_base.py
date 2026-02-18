"""Tests for mada_modelkit._base.

Covers BaseAgentClient: abstract enforcement, concrete subclass instantiation,
send_request signature and delegation, and ABC registration. Further methods
(streaming, generate, health_check, context manager, semaphore) are added as
their respective tasks complete.
"""

from __future__ import annotations

import asyncio

import pytest

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk


class _ConcreteClient(BaseAgentClient):
    """Minimal concrete subclass used by tests."""

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Return a fixed response for any request."""
        return AgentResponse(
            content="ok",
            model="test-model",
            input_tokens=request.max_tokens // 2,
            output_tokens=10,
        )


class TestBaseAgentClientAbstract:
    """Tests that enforce the ABC contract on BaseAgentClient."""

    def test_cannot_instantiate_directly(self) -> None:
        """BaseAgentClient cannot be instantiated without implementing send_request."""
        with pytest.raises(TypeError):
            BaseAgentClient()  # type: ignore[abstract]

    def test_subclass_missing_send_request_cannot_instantiate(self) -> None:
        """A subclass that omits send_request cannot be instantiated."""
        class Incomplete(BaseAgentClient):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_is_abstract_base_class(self) -> None:
        """BaseAgentClient is registered as an ABC."""
        assert hasattr(BaseAgentClient, "__abstractmethods__")

    def test_send_request_is_abstract(self) -> None:
        """send_request is listed in __abstractmethods__."""
        assert "send_request" in BaseAgentClient.__abstractmethods__


class TestConcreteClient:
    """Tests using a minimal concrete implementation of BaseAgentClient."""

    def test_concrete_subclass_instantiates(self) -> None:
        """A subclass that implements send_request can be instantiated."""
        client = _ConcreteClient()
        assert isinstance(client, BaseAgentClient)

    def test_is_instance_of_base(self) -> None:
        """The concrete client is an instance of BaseAgentClient."""
        assert isinstance(_ConcreteClient(), BaseAgentClient)

    @pytest.mark.asyncio
    async def test_send_request_returns_agent_response(self) -> None:
        """send_request returns an AgentResponse."""
        client = _ConcreteClient()
        request = AgentRequest(prompt="hello")
        response = await client.send_request(request)
        assert isinstance(response, AgentResponse)

    @pytest.mark.asyncio
    async def test_send_request_content(self) -> None:
        """send_request returns the expected content."""
        client = _ConcreteClient()
        response = await client.send_request(AgentRequest(prompt="hi"))
        assert response.content == "ok"
        assert response.model == "test-model"

    @pytest.mark.asyncio
    async def test_send_request_receives_full_request(self) -> None:
        """send_request receives and can access all AgentRequest fields."""
        client = _ConcreteClient()
        request = AgentRequest(prompt="test", max_tokens=512)
        response = await client.send_request(request)
        assert response.input_tokens == 256  # max_tokens // 2


class TestSendRequestStreamDefault:
    """Tests for the default send_request_stream implementation."""

    @pytest.mark.asyncio
    async def test_yields_one_chunk(self) -> None:
        """Default stream yields exactly one StreamChunk."""
        client = _ConcreteClient()
        chunks = [chunk async for chunk in client.send_request_stream(AgentRequest(prompt="hi"))]
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_chunk_is_stream_chunk(self) -> None:
        """The yielded item is a StreamChunk instance."""
        client = _ConcreteClient()
        chunks = [chunk async for chunk in client.send_request_stream(AgentRequest(prompt="hi"))]
        assert isinstance(chunks[0], StreamChunk)

    @pytest.mark.asyncio
    async def test_chunk_delta_matches_response_content(self) -> None:
        """The chunk's delta equals the full response content."""
        client = _ConcreteClient()
        chunks = [chunk async for chunk in client.send_request_stream(AgentRequest(prompt="hi"))]
        assert chunks[0].delta == "ok"

    @pytest.mark.asyncio
    async def test_chunk_is_final(self) -> None:
        """The single yielded chunk has is_final=True."""
        client = _ConcreteClient()
        chunks = [chunk async for chunk in client.send_request_stream(AgentRequest(prompt="hi"))]
        assert chunks[0].is_final is True

    @pytest.mark.asyncio
    async def test_stream_delegates_to_send_request(self) -> None:
        """Default stream calls send_request internally to produce the chunk."""
        call_log: list[AgentRequest] = []

        class _TrackingClient(BaseAgentClient):
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                """Record the call and return a fixed response."""
                call_log.append(request)
                return AgentResponse(content="tracked", model="m", input_tokens=1, output_tokens=1)

        client = _TrackingClient()
        request = AgentRequest(prompt="trace me")
        chunks = [chunk async for chunk in client.send_request_stream(request)]
        assert len(call_log) == 1
        assert call_log[0] is request
        assert chunks[0].delta == "tracked"


class TestGenerate:
    """Tests for the generate convenience method."""

    @pytest.mark.asyncio
    async def test_returns_agent_response(self) -> None:
        """generate returns an AgentResponse."""
        client = _ConcreteClient()
        response = await client.generate("hello")
        assert isinstance(response, AgentResponse)

    @pytest.mark.asyncio
    async def test_prompt_passed_through(self) -> None:
        """generate forwards the prompt to send_request via AgentRequest."""
        received: list[AgentRequest] = []

        class _CaptureClient(BaseAgentClient):
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                """Capture the request and return a fixed response."""
                received.append(request)
                return AgentResponse(content="x", model="m", input_tokens=1, output_tokens=1)

        await _CaptureClient().generate("my prompt")
        assert received[0].prompt == "my prompt"

    @pytest.mark.asyncio
    async def test_kwargs_forwarded_to_agent_request(self) -> None:
        """generate passes extra kwargs as AgentRequest fields."""
        received: list[AgentRequest] = []

        class _CaptureClient(BaseAgentClient):
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                """Capture the request and return a fixed response."""
                received.append(request)
                return AgentResponse(content="x", model="m", input_tokens=1, output_tokens=1)

        await _CaptureClient().generate("p", max_tokens=256, temperature=0.1)
        assert received[0].max_tokens == 256
        assert received[0].temperature == 0.1

    @pytest.mark.asyncio
    async def test_delegates_to_send_request(self) -> None:
        """generate calls send_request exactly once."""
        call_count = 0

        class _CountingClient(BaseAgentClient):
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                """Count invocations and return a fixed response."""
                nonlocal call_count
                call_count += 1
                return AgentResponse(content="y", model="m", input_tokens=1, output_tokens=1)

        await _CountingClient().generate("ping")
        assert call_count == 1


class TestGenerateStream:
    """Tests for the generate_stream convenience method."""

    @pytest.mark.asyncio
    async def test_yields_stream_chunks(self) -> None:
        """generate_stream yields StreamChunk instances."""
        client = _ConcreteClient()
        chunks = [c async for c in client.generate_stream("hello")]
        assert all(isinstance(c, StreamChunk) for c in chunks)

    @pytest.mark.asyncio
    async def test_prompt_passed_through(self) -> None:
        """generate_stream forwards the prompt to send_request_stream via AgentRequest."""
        received: list[AgentRequest] = []

        class _CaptureClient(BaseAgentClient):
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                """Capture the request and return a fixed response."""
                received.append(request)
                return AgentResponse(content="z", model="m", input_tokens=1, output_tokens=1)

        chunks = [c async for c in _CaptureClient().generate_stream("stream me")]
        assert received[0].prompt == "stream me"
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_kwargs_forwarded_to_agent_request(self) -> None:
        """generate_stream passes extra kwargs as AgentRequest fields."""
        received: list[AgentRequest] = []

        class _CaptureClient(BaseAgentClient):
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                """Capture the request and return a fixed response."""
                received.append(request)
                return AgentResponse(content="z", model="m", input_tokens=1, output_tokens=1)

        _ = [c async for c in _CaptureClient().generate_stream("p", max_tokens=128)]
        assert received[0].max_tokens == 128

    @pytest.mark.asyncio
    async def test_final_chunk_is_final(self) -> None:
        """The last chunk from generate_stream has is_final=True (via default stream)."""
        client = _ConcreteClient()
        chunks = [c async for c in client.generate_stream("hi")]
        assert chunks[-1].is_final is True


class TestVirtualMethods:
    """Tests for health_check, cancel, and close virtual defaults."""

    @pytest.mark.asyncio
    async def test_health_check_returns_true(self) -> None:
        """Default health_check returns True."""
        assert await _ConcreteClient().health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_return_type_is_bool(self) -> None:
        """Default health_check return value is a bool."""
        result = await _ConcreteClient().health_check()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_cancel_returns_none(self) -> None:
        """Default cancel returns None (no-op)."""
        result = await _ConcreteClient().cancel()
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_does_not_raise(self) -> None:
        """Default cancel completes without raising."""
        await _ConcreteClient().cancel()

    @pytest.mark.asyncio
    async def test_close_returns_none(self) -> None:
        """Default close returns None (no-op)."""
        result = await _ConcreteClient().close()
        assert result is None

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self) -> None:
        """Default close completes without raising."""
        await _ConcreteClient().close()

    @pytest.mark.asyncio
    async def test_health_check_can_be_overridden(self) -> None:
        """A subclass can override health_check to return False."""
        class _UnhealthyClient(BaseAgentClient):
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                """Return a fixed response."""
                return AgentResponse(content="x", model="m", input_tokens=1, output_tokens=1)

            async def health_check(self) -> bool:
                """Always report unhealthy."""
                return False

        assert await _UnhealthyClient().health_check() is False


class TestContextManager:
    """Tests for the async context manager protocol (__aenter__ / __aexit__)."""

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self) -> None:
        """__aenter__ returns the client instance itself."""
        client = _ConcreteClient()
        result = await client.__aenter__()
        assert result is client

    @pytest.mark.asyncio
    async def test_async_with_yields_client(self) -> None:
        """The async with statement binds to the client instance."""
        async with _ConcreteClient() as client:
            assert isinstance(client, _ConcreteClient)

    @pytest.mark.asyncio
    async def test_aexit_calls_close(self) -> None:
        """__aexit__ calls close() on the client."""
        close_calls: list[bool] = []

        class _TrackingClose(BaseAgentClient):
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                """Return a fixed response."""
                return AgentResponse(content="x", model="m", input_tokens=1, output_tokens=1)

            async def close(self) -> None:
                """Record that close was called."""
                close_calls.append(True)

        async with _TrackingClose():
            pass

        assert close_calls == [True]

    @pytest.mark.asyncio
    async def test_close_called_on_exception(self) -> None:
        """close() is called even when the body of the async with raises."""
        close_calls: list[bool] = []

        class _TrackingClose(BaseAgentClient):
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                """Return a fixed response."""
                return AgentResponse(content="x", model="m", input_tokens=1, output_tokens=1)

            async def close(self) -> None:
                """Record that close was called."""
                close_calls.append(True)

        with pytest.raises(RuntimeError):
            async with _TrackingClose():
                raise RuntimeError("body error")

        assert close_calls == [True]

    @pytest.mark.asyncio
    async def test_client_usable_inside_context(self) -> None:
        """send_request is callable normally inside the async with block."""
        async with _ConcreteClient() as client:
            response = await client.send_request(AgentRequest(prompt="hi"))
            assert response.content == "ok"


class TestSemaphore:
    """Tests for max_concurrent / _semaphore initialisation."""

    def test_semaphore_none_by_default(self) -> None:
        """_semaphore is None when max_concurrent is not provided."""
        client = _ConcreteClient()
        assert client._semaphore is None

    def test_semaphore_none_when_explicit_none(self) -> None:
        """_semaphore is None when max_concurrent=None is passed explicitly."""
        client = _ConcreteClient(max_concurrent=None)
        assert client._semaphore is None

    def test_semaphore_created_when_max_concurrent_set(self) -> None:
        """_semaphore is an asyncio.Semaphore when max_concurrent is given."""
        client = _ConcreteClient(max_concurrent=5)
        assert isinstance(client._semaphore, asyncio.Semaphore)

    def test_semaphore_respects_limit(self) -> None:
        """The semaphore's internal counter matches the requested max_concurrent."""
        client = _ConcreteClient(max_concurrent=3)
        assert client._semaphore is not None
        assert client._semaphore._value == 3  # type: ignore[attr-defined]

    def test_semaphore_limit_one(self) -> None:
        """max_concurrent=1 creates a binary semaphore."""
        client = _ConcreteClient(max_concurrent=1)
        assert isinstance(client._semaphore, asyncio.Semaphore)

    def test_semaphore_independent_across_instances(self) -> None:
        """Each instance gets its own semaphore object."""
        c1 = _ConcreteClient(max_concurrent=2)
        c2 = _ConcreteClient(max_concurrent=2)
        assert c1._semaphore is not c2._semaphore
