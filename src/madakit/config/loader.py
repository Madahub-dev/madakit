"""Configuration loader for madakit.

Load configurations from YAML/JSON files, perform environment variable
substitution, and instantiate middleware stacks.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from madakit._base import BaseAgentClient
from madakit.config._schema import (
    MiddlewareConfig,
    ProviderConfig,
    StackConfig,
)

__all__ = ["ConfigLoader", "ConfigError"]


class ConfigError(Exception):
    """Configuration loading or validation error."""

    pass


class ConfigLoader:
    """Load and instantiate middleware stacks from configuration files."""

    @staticmethod
    def _substitute_env_vars(text: str) -> str:
        """Substitute environment variables in text.

        Supports ${VAR_NAME} and ${VAR_NAME:default} syntax.

        Args:
            text: Text containing environment variable references.

        Returns:
            Text with environment variables substituted.

        Raises:
            ConfigError: If a required environment variable is missing.
        """
        # Pattern: ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default_value = match.group(2)

            value = os.environ.get(var_name)

            if value is None:
                if default_value is not None:
                    return default_value
                raise ConfigError(
                    f"Environment variable '{var_name}' is required but not set"
                )

            return value

        return re.sub(pattern, replace_var, text)

    @staticmethod
    def _substitute_env_vars_in_dict(data: dict[str, Any]) -> dict[str, Any]:
        """Recursively substitute environment variables in a dictionary.

        Args:
            data: Dictionary possibly containing env var references.

        Returns:
            Dictionary with environment variables substituted.
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = ConfigLoader._substitute_env_vars(value)
            elif isinstance(value, dict):
                result[key] = ConfigLoader._substitute_env_vars_in_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    ConfigLoader._substitute_env_vars(v) if isinstance(v, str)
                    else ConfigLoader._substitute_env_vars_in_dict(v) if isinstance(v, dict)
                    else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    @staticmethod
    def from_yaml(path: str | Path) -> StackConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            StackConfig instance.

        Raises:
            ConfigError: If YAML parsing fails or config is invalid.
            ImportError: If pyyaml is not installed.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required for YAML configuration. "
                "Install with: pip install pyyaml"
            ) from exc

        try:
            with open(path, 'r') as f:
                raw_data = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise ConfigError(f"Configuration file not found: {path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

        if raw_data is None or not isinstance(raw_data, dict):
            raise ConfigError(f"Configuration file {path} must contain a YAML object")

        return ConfigLoader._parse_config(raw_data)

    @staticmethod
    def from_json(path: str | Path) -> StackConfig:
        """Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file.

        Returns:
            StackConfig instance.

        Raises:
            ConfigError: If JSON parsing fails or config is invalid.
        """
        try:
            with open(path, 'r') as f:
                raw_data = json.load(f)
        except FileNotFoundError as exc:
            raise ConfigError(f"Configuration file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc

        if not isinstance(raw_data, dict):
            raise ConfigError(f"Configuration file {path} must contain a JSON object")

        return ConfigLoader._parse_config(raw_data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> StackConfig:
        """Load configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            StackConfig instance.

        Raises:
            ConfigError: If config is invalid.
        """
        return ConfigLoader._parse_config(data)

    @staticmethod
    def _parse_config(raw_data: dict[str, Any]) -> StackConfig:
        """Parse raw configuration dictionary to StackConfig.

        Args:
            raw_data: Raw configuration data.

        Returns:
            StackConfig instance.

        Raises:
            ConfigError: If config structure is invalid.
        """
        # Substitute environment variables
        try:
            data = ConfigLoader._substitute_env_vars_in_dict(raw_data)
        except ConfigError:
            raise  # Re-raise config errors

        # Validate top-level structure
        if "provider" not in data:
            raise ConfigError("Configuration must contain 'provider' field")

        # Parse provider config
        try:
            provider_data = data["provider"]
            if not isinstance(provider_data, dict):
                raise ConfigError("'provider' must be an object")

            provider_config = ProviderConfig(
                type=provider_data.get("type", ""),
                model=provider_data.get("model"),
                api_key=provider_data.get("api_key"),
                base_url=provider_data.get("base_url"),
                kwargs=provider_data.get("kwargs", {}),
            )
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"Invalid provider configuration: {exc}") from exc

        # Parse middleware configs
        middleware_configs = []
        if "middleware" in data:
            middleware_data = data["middleware"]
            if not isinstance(middleware_data, list):
                raise ConfigError("'middleware' must be a list")

            for i, mw_data in enumerate(middleware_data):
                if not isinstance(mw_data, dict):
                    raise ConfigError(f"middleware[{i}] must be an object")

                try:
                    mw_config = MiddlewareConfig(
                        type=mw_data.get("type", ""),
                        params=mw_data.get("params", {}),
                    )
                    middleware_configs.append(mw_config)
                except (ValueError, TypeError) as exc:
                    raise ConfigError(
                        f"Invalid middleware[{i}] configuration: {exc}"
                    ) from exc

        # Create stack config
        try:
            stack_config = StackConfig(
                provider=provider_config,
                middleware=middleware_configs,
            )
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"Invalid stack configuration: {exc}") from exc

        return stack_config

    @staticmethod
    def build_stack(config: StackConfig) -> BaseAgentClient:
        """Instantiate a middleware stack from configuration.

        Args:
            config: StackConfig specifying the stack to build.

        Returns:
            BaseAgentClient (provider wrapped in middleware).

        Raises:
            ConfigError: If instantiation fails.
        """
        # Import providers and middleware dynamically
        from madakit import middleware as mw_module

        # Provider type mapping
        provider_map = {
            "openai": ("madakit.providers.cloud.openai", "OpenAIClient"),
            "anthropic": ("madakit.providers.cloud.anthropic", "AnthropicClient"),
            "gemini": ("madakit.providers.cloud.gemini", "GeminiClient"),
            "deepseek": ("madakit.providers.cloud.deepseek", "DeepSeekClient"),
            "ollama": ("madakit.providers.local_server.ollama", "OllamaClient"),
            "vllm": ("madakit.providers.local_server.vllm", "VllmClient"),
            "localai": ("madakit.providers.local_server.localai", "LocalAIClient"),
            "llamacpp": ("madakit.providers.native.llamacpp", "LlamaCppClient"),
            "transformers": ("madakit.providers.native.transformers", "TransformersClient"),
        }

        # Middleware type mapping
        middleware_map = {
            "retry": "RetryMiddleware",
            "circuit_breaker": "CircuitBreakerMiddleware",
            "circuit-breaker": "CircuitBreakerMiddleware",
            "cache": "CachingMiddleware",
            "caching": "CachingMiddleware",
            "tracking": "TrackingMiddleware",
            "fallback": "FallbackMiddleware",
            "rate_limit": "RateLimitMiddleware",
            "rate-limit": "RateLimitMiddleware",
            "cost_control": "CostControlMiddleware",
            "cost-control": "CostControlMiddleware",
            "timeout": "TimeoutMiddleware",
            "logging": "LoggingMiddleware",
            "metrics": "MetricsMiddleware",
        }

        # Instantiate provider
        provider_type_lower = config.provider.type.lower()
        if provider_type_lower not in provider_map:
            raise ConfigError(f"Unknown provider type: {config.provider.type}")

        module_path, class_name = provider_map[provider_type_lower]

        try:
            # Import provider module and class
            import importlib
            provider_module = importlib.import_module(module_path)
            provider_class = getattr(provider_module, class_name)

            # Build provider kwargs
            provider_kwargs = {}
            if config.provider.model:
                provider_kwargs["model"] = config.provider.model
            if config.provider.api_key:
                provider_kwargs["api_key"] = config.provider.api_key
            if config.provider.base_url:
                provider_kwargs["base_url"] = config.provider.base_url
            provider_kwargs.update(config.provider.kwargs)

            # Instantiate provider
            client = provider_class(**provider_kwargs)

        except ImportError as exc:
            raise ConfigError(
                f"Failed to import provider {config.provider.type}: {exc}"
            ) from exc
        except Exception as exc:
            raise ConfigError(
                f"Failed to instantiate provider {config.provider.type}: {exc}"
            ) from exc

        # Wrap with middleware (innermost to outermost, reverse order)
        for mw_config in reversed(config.middleware):
            mw_type_lower = mw_config.type.lower()
            if mw_type_lower not in middleware_map:
                raise ConfigError(f"Unknown middleware type: {mw_config.type}")

            mw_class_name = middleware_map[mw_type_lower]

            try:
                # Get middleware class from mw_module
                mw_class = getattr(mw_module, mw_class_name)

                # Instantiate middleware with client and params
                client = mw_class(client=client, **mw_config.params)

            except AttributeError as exc:
                raise ConfigError(
                    f"Middleware {mw_config.type} not found: {exc}"
                ) from exc
            except Exception as exc:
                raise ConfigError(
                    f"Failed to instantiate middleware {mw_config.type}: {exc}"
                ) from exc

        return client
