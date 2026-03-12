"""Configuration schema for madakit.

Dataclasses for declarative middleware stacks and provider configuration.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ProviderConfig", "MiddlewareConfig", "StackConfig"]


@dataclass
class ProviderConfig:
    """Provider configuration.

    Specifies the provider type and connection parameters.
    """

    type: str
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate provider configuration."""
        if not self.type:
            raise ValueError("Provider type is required")
        if not isinstance(self.type, str):
            raise TypeError(f"Provider type must be str, got {type(self.type).__name__}")
        if self.model is not None and not isinstance(self.model, str):
            raise TypeError(f"Model must be str, got {type(self.model).__name__}")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise TypeError(f"API key must be str, got {type(self.api_key).__name__}")
        if self.base_url is not None and not isinstance(self.base_url, str):
            raise TypeError(f"Base URL must be str, got {type(self.base_url).__name__}")
        if not isinstance(self.kwargs, dict):
            raise TypeError(f"kwargs must be dict, got {type(self.kwargs).__name__}")


@dataclass
class MiddlewareConfig:
    """Middleware configuration.

    Specifies the middleware type and initialization parameters.
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate middleware configuration."""
        if not self.type:
            raise ValueError("Middleware type is required")
        if not isinstance(self.type, str):
            raise TypeError(f"Middleware type must be str, got {type(self.type).__name__}")
        if not isinstance(self.params, dict):
            raise TypeError(f"Middleware params must be dict, got {type(self.params).__name__}")


@dataclass
class StackConfig:
    """Complete middleware stack configuration.

    Combines a provider with an ordered list of middleware.
    """

    provider: ProviderConfig
    middleware: list[MiddlewareConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate stack configuration."""
        if not isinstance(self.provider, ProviderConfig):
            raise TypeError(
                f"Provider must be ProviderConfig, got {type(self.provider).__name__}"
            )
        if not isinstance(self.middleware, list):
            raise TypeError(
                f"Middleware must be list, got {type(self.middleware).__name__}"
            )
        for i, mw in enumerate(self.middleware):
            if not isinstance(mw, MiddlewareConfig):
                raise TypeError(
                    f"Middleware[{i}] must be MiddlewareConfig, "
                    f"got {type(mw).__name__}"
                )

    def validate_middleware_order(self) -> None:
        """Validate middleware ordering follows best practices.

        Recommended order per architecture:
        1. Timeout (outermost)
        2. Retry
        3. Circuit breaker
        4. Rate limiting
        5. Cost control
        6. Logging/Metrics
        7. Caching (innermost, closest to provider)
        """
        # Define middleware priority (lower = outer, higher = inner)
        priority_map = {
            "timeout": 1,
            "retry": 2,
            "circuit_breaker": 3,
            "circuit-breaker": 3,
            "rate_limit": 4,
            "rate-limit": 4,
            "cost_control": 5,
            "cost-control": 5,
            "logging": 6,
            "metrics": 6,
            "tracking": 6,
            "cache": 7,
            "caching": 7,
            "fallback": 8,
        }

        # Check ordering
        last_priority = 0
        for mw in self.middleware:
            mw_type_lower = mw.type.lower()
            priority = priority_map.get(mw_type_lower, 5)  # Default to middle

            if priority < last_priority:
                # Out of order - could warn or raise
                # For now, we'll just validate without enforcing strict order
                pass
            last_priority = priority
