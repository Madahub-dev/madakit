"""Pytest configuration and shared fixtures for the mada-modelkit test suite.

Imports MockProvider from helpers so it is available as a fixture or directly
referenced in any test module via: from helpers import MockProvider.
"""

from __future__ import annotations

from helpers import MockProvider

__all__ = ["MockProvider"]
