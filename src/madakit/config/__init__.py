"""Configuration system for madakit.

Declarative middleware stacks via YAML/JSON.
"""

from madakit.config._schema import (
    MiddlewareConfig,
    ProviderConfig,
    StackConfig,
)
from madakit.config.loader import ConfigError, ConfigLoader

__all__ = [
    "ProviderConfig",
    "MiddlewareConfig",
    "StackConfig",
    "ConfigLoader",
    "ConfigError",
]
