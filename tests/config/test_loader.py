"""Tests for configuration loader.

Tests YAML/JSON parsing, environment variable substitution,
stack building, and error handling.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from mada_modelkit.config._schema import (
    MiddlewareConfig,
    ProviderConfig,
    StackConfig,
)
from mada_modelkit.config.loader import ConfigError, ConfigLoader


class TestModuleExports:
    """Test module-level exports."""

    def test_all_exports(self) -> None:
        """__all__ contains loader classes."""
        from mada_modelkit.config import loader

        assert set(loader.__all__) == {"ConfigLoader", "ConfigError"}

    def test_loader_importable(self) -> None:
        """Loader classes importable from config module."""
        from mada_modelkit.config import ConfigError as CE
        from mada_modelkit.config import ConfigLoader as CL

        assert CL is not None
        assert CE is not None


class TestEnvironmentVariableSubstitution:
    """Test environment variable substitution."""

    def test_substitute_simple_var(self) -> None:
        """Simple ${VAR} substitution works."""
        os.environ["TEST_VAR"] = "test_value"
        try:
            result = ConfigLoader._substitute_env_vars("prefix_${TEST_VAR}_suffix")
            assert result == "prefix_test_value_suffix"
        finally:
            del os.environ["TEST_VAR"]

    def test_substitute_multiple_vars(self) -> None:
        """Multiple variables in same string."""
        os.environ["VAR1"] = "value1"
        os.environ["VAR2"] = "value2"
        try:
            result = ConfigLoader._substitute_env_vars("${VAR1} and ${VAR2}")
            assert result == "value1 and value2"
        finally:
            del os.environ["VAR1"]
            del os.environ["VAR2"]

    def test_substitute_with_default(self) -> None:
        """${VAR:default} syntax provides fallback."""
        # Variable not set, should use default
        result = ConfigLoader._substitute_env_vars("${MISSING_VAR:default_value}")
        assert result == "default_value"

    def test_substitute_existing_var_ignores_default(self) -> None:
        """Existing variable takes precedence over default."""
        os.environ["EXISTING_VAR"] = "actual_value"
        try:
            result = ConfigLoader._substitute_env_vars("${EXISTING_VAR:default_value}")
            assert result == "actual_value"
        finally:
            del os.environ["EXISTING_VAR"]

    def test_substitute_missing_var_raises(self) -> None:
        """Missing variable without default raises ConfigError."""
        with pytest.raises(ConfigError, match="Environment variable 'MISSING' is required"):
            ConfigLoader._substitute_env_vars("${MISSING}")

    def test_substitute_in_dict_recursive(self) -> None:
        """Substitution works recursively in nested dicts."""
        os.environ["API_KEY"] = "sk-test"
        try:
            data = {
                "provider": {
                    "api_key": "${API_KEY}",
                    "nested": {
                        "value": "${API_KEY}",
                    },
                },
            }
            result = ConfigLoader._substitute_env_vars_in_dict(data)
            assert result["provider"]["api_key"] == "sk-test"
            assert result["provider"]["nested"]["value"] == "sk-test"
        finally:
            del os.environ["API_KEY"]

    def test_substitute_in_list(self) -> None:
        """Substitution works in lists."""
        os.environ["VALUE"] = "test"
        try:
            data = {"items": ["${VALUE}", "plain", "${VALUE}"]}
            result = ConfigLoader._substitute_env_vars_in_dict(data)
            assert result["items"] == ["test", "plain", "test"]
        finally:
            del os.environ["VALUE"]

    def test_substitute_preserves_non_strings(self) -> None:
        """Non-string values preserved unchanged."""
        data = {
            "number": 42,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
        }
        result = ConfigLoader._substitute_env_vars_in_dict(data)
        assert result == data


class TestYAMLParser:
    """Test YAML configuration parsing."""

    def test_from_yaml_minimal(self) -> None:
        """Parse minimal YAML config."""
        yaml_content = """
provider:
  type: openai
  model: gpt-4
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            config = ConfigLoader.from_yaml(temp_path)
            assert config.provider.type == "openai"
            assert config.provider.model == "gpt-4"
            assert config.middleware == []
        finally:
            os.unlink(temp_path)

    def test_from_yaml_with_middleware(self) -> None:
        """Parse YAML with middleware list."""
        yaml_content = """
provider:
  type: anthropic
  model: claude-3-opus

middleware:
  - type: retry
    params:
      max_retries: 3
  - type: cache
    params:
      ttl: 3600
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            config = ConfigLoader.from_yaml(temp_path)
            assert config.provider.type == "anthropic"
            assert len(config.middleware) == 2
            assert config.middleware[0].type == "retry"
            assert config.middleware[0].params == {"max_retries": 3}
            assert config.middleware[1].type == "cache"
        finally:
            os.unlink(temp_path)

    def test_from_yaml_with_env_vars(self) -> None:
        """Parse YAML with environment variable substitution."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        yaml_content = """
provider:
  type: openai
  api_key: ${OPENAI_API_KEY}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            config = ConfigLoader.from_yaml(temp_path)
            assert config.provider.api_key == "sk-test-key"
        finally:
            os.unlink(temp_path)
            del os.environ["OPENAI_API_KEY"]

    def test_from_yaml_file_not_found(self) -> None:
        """from_yaml raises ConfigError for missing file."""
        with pytest.raises(ConfigError, match="Configuration file not found"):
            ConfigLoader.from_yaml("/nonexistent/path.yaml")

    def test_from_yaml_invalid_yaml(self) -> None:
        """from_yaml raises ConfigError for invalid YAML."""
        yaml_content = """
provider:
  - invalid: syntax
    - broken
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ConfigError, match="Invalid YAML"):
                ConfigLoader.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)

    def test_from_yaml_empty_file(self) -> None:
        """from_yaml raises ConfigError for empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ConfigError, match="must contain a YAML object"):
                ConfigLoader.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)

    def test_from_yaml_requires_pyyaml(self) -> None:
        """from_yaml mentions pyyaml in error when not installed."""
        # This test would only fail if pyyaml not installed
        # Since it's in dev dependencies, we just verify the import works
        try:
            import yaml
            assert yaml is not None
        except ImportError:
            pytest.skip("pyyaml not installed")


class TestJSONParser:
    """Test JSON configuration parsing."""

    def test_from_json_minimal(self) -> None:
        """Parse minimal JSON config."""
        json_content = """
{
  "provider": {
    "type": "openai",
    "model": "gpt-4"
  }
}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            f.flush()
            temp_path = f.name

        try:
            config = ConfigLoader.from_json(temp_path)
            assert config.provider.type == "openai"
            assert config.provider.model == "gpt-4"
            assert config.middleware == []
        finally:
            os.unlink(temp_path)

    def test_from_json_with_middleware(self) -> None:
        """Parse JSON with middleware list."""
        json_content = """
{
  "provider": {
    "type": "anthropic",
    "model": "claude-3-opus"
  },
  "middleware": [
    {
      "type": "retry",
      "params": {
        "max_retries": 3
      }
    },
    {
      "type": "cache",
      "params": {
        "ttl": 3600
      }
    }
  ]
}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            f.flush()
            temp_path = f.name

        try:
            config = ConfigLoader.from_json(temp_path)
            assert config.provider.type == "anthropic"
            assert len(config.middleware) == 2
            assert config.middleware[0].type == "retry"
            assert config.middleware[1].type == "cache"
        finally:
            os.unlink(temp_path)

    def test_from_json_with_env_vars(self) -> None:
        """Parse JSON with environment variable substitution."""
        os.environ["ANTHROPIC_KEY"] = "sk-ant-test"
        json_content = """
{
  "provider": {
    "type": "anthropic",
    "api_key": "${ANTHROPIC_KEY}"
  }
}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            f.flush()
            temp_path = f.name

        try:
            config = ConfigLoader.from_json(temp_path)
            assert config.provider.api_key == "sk-ant-test"
        finally:
            os.unlink(temp_path)
            del os.environ["ANTHROPIC_KEY"]

    def test_from_json_file_not_found(self) -> None:
        """from_json raises ConfigError for missing file."""
        with pytest.raises(ConfigError, match="Configuration file not found"):
            ConfigLoader.from_json("/nonexistent/path.json")

    def test_from_json_invalid_json(self) -> None:
        """from_json raises ConfigError for invalid JSON."""
        json_content = "{ invalid json }"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ConfigError, match="Invalid JSON"):
                ConfigLoader.from_json(temp_path)
        finally:
            os.unlink(temp_path)

    def test_from_json_non_object_root(self) -> None:
        """from_json raises ConfigError for non-object root."""
        json_content = '["array", "root"]'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ConfigError, match="must contain a JSON object"):
                ConfigLoader.from_json(temp_path)
        finally:
            os.unlink(temp_path)


class TestDictParser:
    """Test dictionary configuration parsing."""

    def test_from_dict_minimal(self) -> None:
        """Parse minimal dict config."""
        data = {
            "provider": {
                "type": "openai",
                "model": "gpt-4",
            }
        }
        config = ConfigLoader.from_dict(data)
        assert config.provider.type == "openai"
        assert config.provider.model == "gpt-4"

    def test_from_dict_with_env_vars(self) -> None:
        """Parse dict with environment variable substitution."""
        os.environ["TEST_KEY"] = "key-value"
        try:
            data = {
                "provider": {
                    "type": "openai",
                    "api_key": "${TEST_KEY}",
                }
            }
            config = ConfigLoader.from_dict(data)
            assert config.provider.api_key == "key-value"
        finally:
            del os.environ["TEST_KEY"]

    def test_from_dict_missing_provider(self) -> None:
        """from_dict raises ConfigError when provider missing."""
        data = {"middleware": []}
        with pytest.raises(ConfigError, match="must contain 'provider' field"):
            ConfigLoader.from_dict(data)

    def test_from_dict_invalid_provider_type(self) -> None:
        """from_dict raises ConfigError for invalid provider structure."""
        data = {"provider": "invalid"}
        with pytest.raises(ConfigError, match="'provider' must be an object"):
            ConfigLoader.from_dict(data)

    def test_from_dict_invalid_middleware_type(self) -> None:
        """from_dict raises ConfigError for invalid middleware structure."""
        data = {
            "provider": {"type": "openai"},
            "middleware": "invalid"
        }
        with pytest.raises(ConfigError, match="'middleware' must be a list"):
            ConfigLoader.from_dict(data)

    def test_from_dict_invalid_middleware_item(self) -> None:
        """from_dict raises ConfigError for invalid middleware item."""
        data = {
            "provider": {"type": "openai"},
            "middleware": ["invalid"]
        }
        with pytest.raises(ConfigError, match="middleware\\[0\\] must be an object"):
            ConfigLoader.from_dict(data)


class TestStackBuilder:
    """Test stack building from configuration."""

    def test_build_stack_openai_provider(self) -> None:
        """Build stack with OpenAI provider."""
        config = StackConfig(
            provider=ProviderConfig(
                type="openai",
                model="gpt-4",
                api_key="sk-test",
            )
        )

        # Should not raise (but will fail without httpx installed)
        try:
            client = ConfigLoader.build_stack(config)
            from mada_modelkit.providers.cloud.openai import OpenAIClient
            assert isinstance(client, OpenAIClient)
        except ImportError:
            pytest.skip("httpx not installed")

    def test_build_stack_with_retry_middleware(self) -> None:
        """Build stack with retry middleware."""
        config = StackConfig(
            provider=ProviderConfig(type="openai", model="gpt-4", api_key="sk-test"),
            middleware=[
                MiddlewareConfig(type="retry", params={"max_retries": 3})
            ]
        )

        try:
            client = ConfigLoader.build_stack(config)
            from mada_modelkit.middleware.retry import RetryMiddleware
            assert isinstance(client, RetryMiddleware)
        except ImportError:
            pytest.skip("httpx not installed")

    def test_build_stack_multiple_middleware(self) -> None:
        """Build stack with multiple middleware layers."""
        config = StackConfig(
            provider=ProviderConfig(type="openai", model="gpt-4", api_key="sk-test"),
            middleware=[
                MiddlewareConfig(type="retry", params={"max_retries": 3}),
                MiddlewareConfig(type="cache", params={"ttl": 3600}),
            ]
        )

        try:
            client = ConfigLoader.build_stack(config)
            from mada_modelkit.middleware.retry import RetryMiddleware
            # Middleware list is reversed, so retry (first in list) becomes innermost,
            # cache (last) becomes outermost, wrapping retry
            # But we reversed again in build_stack, so: provider → cache → retry
            # Outermost is retry
            assert isinstance(client, RetryMiddleware)
        except ImportError:
            pytest.skip("httpx not installed")

    def test_build_stack_unknown_provider_raises(self) -> None:
        """build_stack raises ConfigError for unknown provider."""
        config = StackConfig(
            provider=ProviderConfig(type="unknown_provider")
        )

        with pytest.raises(ConfigError, match="Unknown provider type"):
            ConfigLoader.build_stack(config)

    def test_build_stack_unknown_middleware_raises(self) -> None:
        """build_stack raises ConfigError for unknown middleware."""
        config = StackConfig(
            provider=ProviderConfig(type="openai", api_key="sk-test"),
            middleware=[MiddlewareConfig(type="unknown_middleware")]
        )

        with pytest.raises(ConfigError, match="Unknown middleware type"):
            ConfigLoader.build_stack(config)


class TestErrorHandling:
    """Test error handling and messages."""

    def test_config_error_is_exception(self) -> None:
        """ConfigError is an Exception."""
        assert issubclass(ConfigError, Exception)

    def test_config_error_with_message(self) -> None:
        """ConfigError accepts and preserves message."""
        err = ConfigError("Test error message")
        assert str(err) == "Test error message"

    def test_missing_env_var_clear_error(self) -> None:
        """Missing environment variable produces clear error."""
        with pytest.raises(ConfigError, match="Environment variable 'MISSING_VAR' is required"):
            ConfigLoader._substitute_env_vars("${MISSING_VAR}")

    def test_invalid_provider_clear_error(self) -> None:
        """Invalid provider config produces clear error."""
        data = {"provider": {"type": ""}}  # Empty type
        with pytest.raises(ConfigError, match="Invalid provider configuration"):
            ConfigLoader.from_dict(data)

    def test_invalid_middleware_clear_error(self) -> None:
        """Invalid middleware config produces clear error."""
        data = {
            "provider": {"type": "openai"},
            "middleware": [{"type": ""}]  # Empty type
        }
        with pytest.raises(ConfigError, match="Invalid middleware\\[0\\] configuration"):
            ConfigLoader.from_dict(data)
