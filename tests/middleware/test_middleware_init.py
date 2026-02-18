"""Tests for the middleware package __init__ (task 2.6.1).

Covers: all five middleware classes re-exported from the package top-level,
__all__ completeness, and identity of each re-export with its source module.
"""

from __future__ import annotations

import mada_modelkit.middleware as middleware_pkg
from mada_modelkit.middleware import (
    CachingMiddleware,
    CircuitBreakerMiddleware,
    FallbackMiddleware,
    RetryMiddleware,
    TrackingMiddleware,
)


class TestMiddlewarePackageExports:
    """middleware/__init__ — __all__ completeness and re-export identity."""

    def test_all_is_defined(self) -> None:
        """Asserts that __all__ is defined on the middleware package."""
        assert hasattr(middleware_pkg, "__all__")

    def test_all_contains_retry_middleware(self) -> None:
        """Asserts that 'RetryMiddleware' is listed in __all__."""
        assert "RetryMiddleware" in middleware_pkg.__all__

    def test_all_contains_circuit_breaker_middleware(self) -> None:
        """Asserts that 'CircuitBreakerMiddleware' is listed in __all__."""
        assert "CircuitBreakerMiddleware" in middleware_pkg.__all__

    def test_all_contains_caching_middleware(self) -> None:
        """Asserts that 'CachingMiddleware' is listed in __all__."""
        assert "CachingMiddleware" in middleware_pkg.__all__

    def test_all_contains_tracking_middleware(self) -> None:
        """Asserts that 'TrackingMiddleware' is listed in __all__."""
        assert "TrackingMiddleware" in middleware_pkg.__all__

    def test_all_contains_fallback_middleware(self) -> None:
        """Asserts that 'FallbackMiddleware' is listed in __all__."""
        assert "FallbackMiddleware" in middleware_pkg.__all__

    def test_retry_middleware_is_same_class_as_source(self) -> None:
        """Asserts that RetryMiddleware re-export is identical to its source class."""
        from mada_modelkit.middleware.retry import RetryMiddleware as Source
        assert RetryMiddleware is Source

    def test_circuit_breaker_middleware_is_same_class_as_source(self) -> None:
        """Asserts that CircuitBreakerMiddleware re-export is identical to its source class."""
        from mada_modelkit.middleware.circuit_breaker import CircuitBreakerMiddleware as Source
        assert CircuitBreakerMiddleware is Source

    def test_caching_middleware_is_same_class_as_source(self) -> None:
        """Asserts that CachingMiddleware re-export is identical to its source class."""
        from mada_modelkit.middleware.cache import CachingMiddleware as Source
        assert CachingMiddleware is Source

    def test_tracking_middleware_is_same_class_as_source(self) -> None:
        """Asserts that TrackingMiddleware re-export is identical to its source class."""
        from mada_modelkit.middleware.tracking import TrackingMiddleware as Source
        assert TrackingMiddleware is Source

    def test_fallback_middleware_is_same_class_as_source(self) -> None:
        """Asserts that FallbackMiddleware re-export is identical to its source class."""
        from mada_modelkit.middleware.fallback import FallbackMiddleware as Source
        assert FallbackMiddleware is Source

    def test_all_has_exactly_five_entries(self) -> None:
        """Asserts that __all__ contains exactly the five expected middleware names."""
        expected = {
            "RetryMiddleware",
            "CircuitBreakerMiddleware",
            "CachingMiddleware",
            "TrackingMiddleware",
            "FallbackMiddleware",
        }
        assert set(middleware_pkg.__all__) == expected
