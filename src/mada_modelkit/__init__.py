"""mada-modelkit — composable AI client library.

Re-exports the public API surface: base client, types, errors, and middleware.
"""

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._errors import (
    AgentError,
    CircuitOpenError,
    MiddlewareError,
    ProviderError,
    RetryExhaustedError,
)
from mada_modelkit._types import (
    AgentRequest,
    AgentResponse,
    Attachment,
    StreamChunk,
    TrackingStats,
)
from mada_modelkit.middleware import (
    ABTestMiddleware,
    CachingMiddleware,
    CircuitBreakerMiddleware,
    ContentFilterMiddleware,
    FallbackMiddleware,
    RetryMiddleware,
    TrackingMiddleware,
)

__all__ = [
    "BaseAgentClient",
    "Attachment",
    "AgentRequest",
    "AgentResponse",
    "StreamChunk",
    "TrackingStats",
    "AgentError",
    "ProviderError",
    "CircuitOpenError",
    "RetryExhaustedError",
    "MiddlewareError",
    "ABTestMiddleware",
    "RetryMiddleware",
    "CircuitBreakerMiddleware",
    "CachingMiddleware",
    "ContentFilterMiddleware",
    "TrackingMiddleware",
    "FallbackMiddleware",
]
