"""Tests for mada_modelkit._base.

Covers BaseAgentClient: abstract enforcement, concrete subclass instantiation,
send_request signature and delegation, and ABC registration. Further methods
(streaming, generate, health_check, context manager, semaphore) are added as
their respective tasks complete.
"""

from __future__ import annotations

import pytest

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse


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
