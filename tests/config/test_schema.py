"""Tests for configuration schema.

Tests ProviderConfig, MiddlewareConfig, StackConfig dataclasses,
validation, and type checking.
"""

from __future__ import annotations

import pytest

from mada_modelkit.config._schema import (
    MiddlewareConfig,
    ProviderConfig,
    StackConfig,
)


class TestModuleExports:
    """Test module-level exports and imports."""

    def test_all_exports(self) -> None:
        """__all__ contains config classes."""
        from mada_modelkit.config import _schema

        assert set(_schema.__all__) == {"ProviderConfig", "MiddlewareConfig", "StackConfig"}

    def test_schema_importable(self) -> None:
        """Schema classes importable from config module."""
        from mada_modelkit.config import (
            MiddlewareConfig as MC,
            ProviderConfig as PC,
            StackConfig as SC,
        )

        assert PC is not None
        assert MC is not None
        assert SC is not None


class TestProviderConfig:
    """Test ProviderConfig dataclass."""

    def test_minimal_constructor(self) -> None:
        """Constructor accepts only type."""
        config = ProviderConfig(type="openai")

        assert config.type == "openai"
        assert config.model is None
        assert config.api_key is None
        assert config.base_url is None
        assert config.kwargs == {}

    def test_all_fields_populated(self) -> None:
        """Constructor accepts all fields."""
        config = ProviderConfig(
            type="anthropic",
            model="claude-3-opus",
            api_key="sk-test",
            base_url="https://api.anthropic.com",
            kwargs={"timeout": 30},
        )

        assert config.type == "anthropic"
        assert config.model == "claude-3-opus"
        assert config.api_key == "sk-test"
        assert config.base_url == "https://api.anthropic.com"
        assert config.kwargs == {"timeout": 30}

    def test_kwargs_defaults_to_empty_dict(self) -> None:
        """kwargs defaults to empty dict."""
        config = ProviderConfig(type="openai")

        assert config.kwargs == {}
        assert isinstance(config.kwargs, dict)

    def test_type_required(self) -> None:
        """Type field is required."""
        with pytest.raises(ValueError, match="Provider type is required"):
            ProviderConfig(type="")

    def test_type_must_be_string(self) -> None:
        """Type must be string."""
        with pytest.raises(TypeError, match="Provider type must be str"):
            ProviderConfig(type=123)  # type: ignore

    def test_model_must_be_string_or_none(self) -> None:
        """Model must be string or None."""
        # Valid
        ProviderConfig(type="openai", model="gpt-4")
        ProviderConfig(type="openai", model=None)

        # Invalid
        with pytest.raises(TypeError, match="Model must be str"):
            ProviderConfig(type="openai", model=123)  # type: ignore

    def test_api_key_must_be_string_or_none(self) -> None:
        """API key must be string or None."""
        # Valid
        ProviderConfig(type="openai", api_key="sk-test")
        ProviderConfig(type="openai", api_key=None)

        # Invalid
        with pytest.raises(TypeError, match="API key must be str"):
            ProviderConfig(type="openai", api_key=123)  # type: ignore

    def test_base_url_must_be_string_or_none(self) -> None:
        """Base URL must be string or None."""
        # Valid
        ProviderConfig(type="openai", base_url="https://api.openai.com")
        ProviderConfig(type="openai", base_url=None)

        # Invalid
        with pytest.raises(TypeError, match="Base URL must be str"):
            ProviderConfig(type="openai", base_url=123)  # type: ignore

    def test_kwargs_must_be_dict(self) -> None:
        """kwargs must be dict."""
        with pytest.raises(TypeError, match="kwargs must be dict"):
            ProviderConfig(type="openai", kwargs="invalid")  # type: ignore

    def test_various_provider_types(self) -> None:
        """Different provider types accepted."""
        providers = ["openai", "anthropic", "gemini", "ollama", "llamacpp"]

        for ptype in providers:
            config = ProviderConfig(type=ptype)
            assert config.type == ptype


class TestMiddlewareConfig:
    """Test MiddlewareConfig dataclass."""

    def test_minimal_constructor(self) -> None:
        """Constructor accepts only type."""
        config = MiddlewareConfig(type="retry")

        assert config.type == "retry"
        assert config.params == {}

    def test_with_params(self) -> None:
        """Constructor accepts params."""
        config = MiddlewareConfig(
            type="retry",
            params={"max_retries": 3, "backoff_factor": 2.0}
        )

        assert config.type == "retry"
        assert config.params == {"max_retries": 3, "backoff_factor": 2.0}

    def test_params_defaults_to_empty_dict(self) -> None:
        """params defaults to empty dict."""
        config = MiddlewareConfig(type="cache")

        assert config.params == {}
        assert isinstance(config.params, dict)

    def test_type_required(self) -> None:
        """Type field is required."""
        with pytest.raises(ValueError, match="Middleware type is required"):
            MiddlewareConfig(type="")

    def test_type_must_be_string(self) -> None:
        """Type must be string."""
        with pytest.raises(TypeError, match="Middleware type must be str"):
            MiddlewareConfig(type=123)  # type: ignore

    def test_params_must_be_dict(self) -> None:
        """params must be dict."""
        with pytest.raises(TypeError, match="Middleware params must be dict"):
            MiddlewareConfig(type="retry", params="invalid")  # type: ignore

    def test_various_middleware_types(self) -> None:
        """Different middleware types accepted."""
        middleware = ["retry", "circuit_breaker", "cache", "logging", "metrics"]

        for mtype in middleware:
            config = MiddlewareConfig(type=mtype)
            assert config.type == mtype


class TestStackConfig:
    """Test StackConfig dataclass."""

    def test_minimal_constructor(self) -> None:
        """Constructor accepts only provider."""
        provider = ProviderConfig(type="openai")
        stack = StackConfig(provider=provider)

        assert stack.provider is provider
        assert stack.middleware == []

    def test_with_middleware_list(self) -> None:
        """Constructor accepts middleware list."""
        provider = ProviderConfig(type="openai")
        middleware = [
            MiddlewareConfig(type="retry"),
            MiddlewareConfig(type="cache"),
        ]
        stack = StackConfig(provider=provider, middleware=middleware)

        assert stack.provider is provider
        assert len(stack.middleware) == 2
        assert stack.middleware[0].type == "retry"
        assert stack.middleware[1].type == "cache"

    def test_middleware_defaults_to_empty_list(self) -> None:
        """middleware defaults to empty list."""
        provider = ProviderConfig(type="openai")
        stack = StackConfig(provider=provider)

        assert stack.middleware == []
        assert isinstance(stack.middleware, list)

    def test_provider_must_be_provider_config(self) -> None:
        """Provider must be ProviderConfig instance."""
        with pytest.raises(TypeError, match="Provider must be ProviderConfig"):
            StackConfig(provider="invalid")  # type: ignore

    def test_middleware_must_be_list(self) -> None:
        """Middleware must be list."""
        provider = ProviderConfig(type="openai")

        with pytest.raises(TypeError, match="Middleware must be list"):
            StackConfig(provider=provider, middleware="invalid")  # type: ignore

    def test_middleware_items_must_be_middleware_config(self) -> None:
        """Middleware list items must be MiddlewareConfig."""
        provider = ProviderConfig(type="openai")

        with pytest.raises(TypeError, match="Middleware\\[0\\] must be MiddlewareConfig"):
            StackConfig(provider=provider, middleware=["invalid"])  # type: ignore

    def test_full_stack_construction(self) -> None:
        """Complete stack with provider and multiple middleware."""
        provider = ProviderConfig(
            type="anthropic",
            model="claude-3-opus",
            api_key="sk-test",
        )
        middleware = [
            MiddlewareConfig(type="retry", params={"max_retries": 3}),
            MiddlewareConfig(type="circuit_breaker", params={"failure_threshold": 5}),
            MiddlewareConfig(type="cache", params={"ttl": 3600}),
        ]
        stack = StackConfig(provider=provider, middleware=middleware)

        assert stack.provider.type == "anthropic"
        assert len(stack.middleware) == 3
        assert stack.middleware[0].type == "retry"
        assert stack.middleware[1].type == "circuit_breaker"
        assert stack.middleware[2].type == "cache"


class TestMiddlewareOrderValidation:
    """Test middleware order validation."""

    def test_validate_middleware_order_exists(self) -> None:
        """validate_middleware_order method exists."""
        provider = ProviderConfig(type="openai")
        stack = StackConfig(provider=provider)

        assert hasattr(stack, "validate_middleware_order")
        assert callable(stack.validate_middleware_order)

    def test_validate_middleware_order_runs_without_error(self) -> None:
        """validate_middleware_order runs without raising."""
        provider = ProviderConfig(type="openai")
        middleware = [
            MiddlewareConfig(type="timeout"),
            MiddlewareConfig(type="retry"),
            MiddlewareConfig(type="cache"),
        ]
        stack = StackConfig(provider=provider, middleware=middleware)

        # Should not raise
        stack.validate_middleware_order()

    def test_empty_middleware_validates(self) -> None:
        """Empty middleware list validates successfully."""
        provider = ProviderConfig(type="openai")
        stack = StackConfig(provider=provider)

        stack.validate_middleware_order()

    def test_single_middleware_validates(self) -> None:
        """Single middleware validates successfully."""
        provider = ProviderConfig(type="openai")
        stack = StackConfig(provider=provider, middleware=[MiddlewareConfig(type="retry")])

        stack.validate_middleware_order()

    def test_optimal_order_validates(self) -> None:
        """Optimal middleware order validates."""
        provider = ProviderConfig(type="openai")
        middleware = [
            MiddlewareConfig(type="timeout"),
            MiddlewareConfig(type="retry"),
            MiddlewareConfig(type="circuit_breaker"),
            MiddlewareConfig(type="rate_limit"),
            MiddlewareConfig(type="cost_control"),
            MiddlewareConfig(type="logging"),
            MiddlewareConfig(type="cache"),
        ]
        stack = StackConfig(provider=provider, middleware=middleware)

        stack.validate_middleware_order()

    def test_out_of_order_middleware_still_validates(self) -> None:
        """Out of order middleware doesn't raise (validation is advisory)."""
        provider = ProviderConfig(type="openai")
        middleware = [
            MiddlewareConfig(type="cache"),  # Should be last
            MiddlewareConfig(type="retry"),  # Should be early
        ]
        stack = StackConfig(provider=provider, middleware=middleware)

        # Should not raise - validation is advisory, not enforced
        stack.validate_middleware_order()
