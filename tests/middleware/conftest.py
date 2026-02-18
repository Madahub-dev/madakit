"""Pytest configuration for the middleware test sub-suite.

Imports MockProvider from helpers so it is available across all middleware
test modules via: from helpers import MockProvider.
"""

from __future__ import annotations

from helpers import MockProvider

__all__ = ["MockProvider"]
