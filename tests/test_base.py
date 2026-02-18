"""Tests for mada_modelkit._base.

Covers BaseAgentClient: abstract enforcement, concrete subclass instantiation,
send_request signature and delegation, and ABC registration. Further methods
(streaming, generate, health_check, context manager, semaphore) are added as
their respective tasks complete.
"""

from __future__ import annotations

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
