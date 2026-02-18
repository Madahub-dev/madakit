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
