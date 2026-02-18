from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Attachment:
    """Binary content attached to a request (image, PDF, etc.)."""

    content: bytes
    media_type: str
    filename: str | None = None


@dataclass
class AgentRequest:
    """Immutable description of what to send to a model."""

    prompt: str
    system_prompt: str | None = None
    attachments: list[Attachment] = field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.7
    stop: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """What came back from the model."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class StreamChunk:
    """A single chunk yielded during a streaming response."""

    delta: str
    is_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackingStats:
    """Aggregate statistics collected by TrackingMiddleware."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_inference_ms: float = 0.0
    total_ttft_ms: float = 0.0
    total_cost_usd: float | None = None

    def reset(self) -> TrackingStats:
        """Return a snapshot of current stats, then zero all counters."""
        snapshot = TrackingStats(
            total_requests=self.total_requests,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_inference_ms=self.total_inference_ms,
            total_ttft_ms=self.total_ttft_ms,
            total_cost_usd=self.total_cost_usd,
        )
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_inference_ms = 0.0
        self.total_ttft_ms = 0.0
        self.total_cost_usd = None
        return snapshot
