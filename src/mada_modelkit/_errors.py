"""Error hierarchy for mada-modelkit.

Defines all exceptions raised by the library. Single root (AgentError),
two branches: ProviderError for backend failures and MiddlewareError for
cross-cutting concerns. No external dependencies — stdlib only.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all mada-modelkit errors."""


class ProviderError(AgentError):
    """Raised when a backend provider returns an error or is unreachable."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Store the error message and optional HTTP status code."""
        super().__init__(message)
        self.status_code = status_code


class MiddlewareError(AgentError):
    """Base class for errors raised by middleware layers."""


class CircuitOpenError(MiddlewareError):
    """Raised when a request is rejected because the circuit breaker is open."""


class RetryExhaustedError(MiddlewareError):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, last_error: Exception) -> None:
        """Store the error message and the exception from the final attempt."""
        super().__init__(message)
        self.last_error = last_error


class BudgetExceededError(MiddlewareError):
    """Raised when a request would exceed the configured budget cap."""
