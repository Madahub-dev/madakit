"""madakit — composable AI client library.

Re-exports the public API surface: base client, types, errors, and middleware.
"""

from madakit._base import BaseAgentClient
from madakit._errors import (
    AgentError,
    CircuitOpenError,
    MiddlewareError,
    ProviderError,
    RetryExhaustedError,
)
from madakit._types import (
    AgentRequest,
    AgentResponse,
    Attachment,
    StreamChunk,
    TrackingStats,
)
from madakit.middleware import (
    ABTestMiddleware,
    BatchingMiddleware,
    CachingMiddleware,
    CircuitBreakerMiddleware,
    ConsensusMiddleware,
    ContentFilterMiddleware,
    FallbackMiddleware,
    FunctionCallingMiddleware,
    LoadBalancingMiddleware,
    PromptTemplateMiddleware,
    RetryMiddleware,
    StreamAggregationMiddleware,
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
    "BatchingMiddleware",
    "RetryMiddleware",
    "CircuitBreakerMiddleware",
    "CachingMiddleware",
    "ConsensusMiddleware",
    "ContentFilterMiddleware",
    "FunctionCallingMiddleware",
    "LoadBalancingMiddleware",
    "PromptTemplateMiddleware",
    "StreamAggregationMiddleware",
    "TrackingMiddleware",
    "FallbackMiddleware",
]
