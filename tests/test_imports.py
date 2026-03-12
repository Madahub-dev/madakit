"""Tests that the public API surface is importable and complete."""

from __future__ import annotations

import pytest


class TestCoreImports:
    """Verify top-level package exports work with zero external dependencies."""

    def test_import_package(self) -> None:
        import madakit

        assert hasattr(madakit, "__all__")

    def test_all_has_21_names(self) -> None:
        from madakit import __all__

        assert len(__all__) == 21

    @pytest.mark.parametrize(
        "name",
        [
            "BaseAgentClient",
            "Attachment",
            "AgentRequest",
            "AgentResponse",
            "StreamChunk",
            "TrackingStats",
            "AgentError",
            "ProviderError",
            "CircuitOpenError",
            "RetryExhaustedError",
            "MiddlewareError",
            "ABTestMiddleware",
            "RetryMiddleware",
            "CircuitBreakerMiddleware",
            "CachingMiddleware",
            "ContentFilterMiddleware",
            "FunctionCallingMiddleware",
            "LoadBalancingMiddleware",
            "PromptTemplateMiddleware",
            "TrackingMiddleware",
            "FallbackMiddleware",
        ],
    )
    def test_public_name_importable(self, name: str) -> None:
        import madakit

        assert hasattr(madakit, name)
        assert name in madakit.__all__

    def test_base_client_is_abc(self) -> None:
        from madakit import BaseAgentClient
        import abc

        assert issubclass(BaseAgentClient, abc.ABC)

    def test_error_hierarchy(self) -> None:
        from madakit import (
            AgentError,
            ProviderError,
            MiddlewareError,
            CircuitOpenError,
            RetryExhaustedError,
        )

        assert issubclass(ProviderError, AgentError)
        assert issubclass(MiddlewareError, AgentError)
        assert issubclass(CircuitOpenError, MiddlewareError)
        assert issubclass(RetryExhaustedError, MiddlewareError)

    def test_middleware_are_base_clients(self) -> None:
        from madakit import (
            BaseAgentClient,
            ABTestMiddleware,
            RetryMiddleware,
            CircuitBreakerMiddleware,
            CachingMiddleware,
            ContentFilterMiddleware,
            FunctionCallingMiddleware,
            LoadBalancingMiddleware,
            PromptTemplateMiddleware,
            TrackingMiddleware,
            FallbackMiddleware,
        )

        for cls in [
            ABTestMiddleware,
            RetryMiddleware,
            CircuitBreakerMiddleware,
            CachingMiddleware,
            ContentFilterMiddleware,
            FunctionCallingMiddleware,
            LoadBalancingMiddleware,
            PromptTemplateMiddleware,
            TrackingMiddleware,
            FallbackMiddleware,
        ]:
            assert issubclass(cls, BaseAgentClient)


class TestProviderImportIsolation:
    """Verify provider subpackages don't fail on import of the module itself."""

    def test_providers_package_imports(self) -> None:
        import madakit.providers

        assert madakit.providers is not None

    def test_cloud_package_imports(self) -> None:
        import madakit.providers.cloud

        assert madakit.providers.cloud is not None

    def test_local_server_package_imports(self) -> None:
        import madakit.providers.local_server

        assert madakit.providers.local_server is not None

    def test_native_package_imports(self) -> None:
        import madakit.providers.native

        assert madakit.providers.native is not None
