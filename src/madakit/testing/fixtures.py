"""Pytest fixtures for mada-modelkit testing.

Common fixtures for testing middleware and providers.
"""

from __future__ import annotations

import pytest

from madakit._types import AgentRequest, AgentResponse
from madakit.testing.utils import MockProvider

__all__ = [
    "mock_provider",
    "sample_request",
    "sample_response",
]


@pytest.fixture
def mock_provider() -> MockProvider:
    """Provide a basic MockProvider instance.

    Returns:
        MockProvider with default configuration.
    """
    return MockProvider()


@pytest.fixture
def sample_request() -> AgentRequest:
    """Provide a sample AgentRequest.

    Returns:
        AgentRequest with common test values.
    """
    return AgentRequest(
        prompt="Test prompt",
        system_prompt="You are helpful",
        max_tokens=100,
        temperature=0.7,
    )


@pytest.fixture
def sample_response() -> AgentResponse:
    """Provide a sample AgentResponse.

    Returns:
        AgentResponse with common test values.
    """
    return AgentResponse(
        content="Test response",
        model="test-model",
        input_tokens=10,
        output_tokens=20,
    )
