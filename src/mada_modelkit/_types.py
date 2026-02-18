from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Attachment:
    """Binary content attached to a request (image, PDF, etc.)."""

    content: bytes
    media_type: str
    filename: str | None = None
