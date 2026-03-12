"""Middleware package for mada-modelkit.

Re-exports all middleware classes for convenient top-level access.
Zero external dependencies — stdlib only.
"""

from madakit.middleware.ab_test import ABTestMiddleware
from madakit.middleware.batching import BatchingMiddleware
from madakit.middleware.cache import CachingMiddleware
from madakit.middleware.circuit_breaker import CircuitBreakerMiddleware
from madakit.middleware.consensus import ConsensusMiddleware
from madakit.middleware.content_filter import ContentFilterMiddleware
from madakit.middleware.fallback import FallbackMiddleware
from madakit.middleware.function_calling import FunctionCallingMiddleware
from madakit.middleware.load_balancing import LoadBalancingMiddleware
from madakit.middleware.prompt_template import PromptTemplateMiddleware
from madakit.middleware.retry import RetryMiddleware
from madakit.middleware.stream_aggregation import StreamAggregationMiddleware
from madakit.middleware.tracking import TrackingMiddleware

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
