"""Configuration system for madakit.

Declarative middleware stacks via YAML/JSON.
"""

from mada_modelkit.config._schema import (
    MiddlewareConfig,
    ProviderConfig,
    StackConfig,
)
from mada_modelkit.config.loader import ConfigError, ConfigLoader

__all__ = [
    "ProviderConfig",
    "MiddlewareConfig",
    "StackConfig",
    "ConfigLoader",
    "ConfigError",
]
