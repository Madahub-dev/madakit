"""Abstract base class for all mada-modelkit providers and middleware.

Defines BaseAgentClient — the single interface every provider and middleware
implements. One abstract method (send_request); all other methods are virtual
with sensible defaults. No external dependencies — stdlib only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from mada_modelkit._types import AgentRequest, AgentResponse


class BaseAgentClient(ABC):
    """Common interface for every AI backend and middleware layer."""

    @abstractmethod
    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute a request and return a complete response."""
        ...
