"""Middleware package for mada-modelkit.

Re-exports all middleware classes for convenient top-level access.
Zero external dependencies — stdlib only.
"""

from mada_modelkit.middleware.ab_test import ABTestMiddleware
from mada_modelkit.middleware.batching import BatchingMiddleware
from mada_modelkit.middleware.cache import CachingMiddleware
from mada_modelkit.middleware.circuit_breaker import CircuitBreakerMiddleware
from mada_modelkit.middleware.consensus import ConsensusMiddleware
from mada_modelkit.middleware.content_filter import ContentFilterMiddleware
from mada_modelkit.middleware.fallback import FallbackMiddleware
from mada_modelkit.middleware.function_calling import FunctionCallingMiddleware
from mada_modelkit.middleware.load_balancing import LoadBalancingMiddleware
from mada_modelkit.middleware.prompt_template import PromptTemplateMiddleware
from mada_modelkit.middleware.retry import RetryMiddleware
from mada_modelkit.middleware.stream_aggregation import StreamAggregationMiddleware
from mada_modelkit.middleware.tracking import TrackingMiddleware

__all__ = [
    "ABTestMiddleware",
    "BatchingMiddleware",
    "CachingMiddleware",
    "CircuitBreakerMiddleware",
    "ConsensusMiddleware",
    "ContentFilterMiddleware",
    "FallbackMiddleware",
    "FunctionCallingMiddleware",
    "LoadBalancingMiddleware",
    "PromptTemplateMiddleware",
    "RetryMiddleware",
    "StreamAggregationMiddleware",
    "TrackingMiddleware",
]
