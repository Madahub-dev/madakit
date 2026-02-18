"""Error hierarchy for mada-modelkit.

Defines all exceptions raised by the library. Single root (AgentError),
two branches: ProviderError for backend failures and MiddlewareError for
cross-cutting concerns. No external dependencies — stdlib only.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all mada-modelkit errors."""
