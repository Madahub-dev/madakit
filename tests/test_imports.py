"""Tests that the public API surface is importable and complete."""

from __future__ import annotations

import pytest


class TestCoreImports:
    """Verify top-level package exports work with zero external dependencies."""

    def test_import_package(self) -> None:
        import mada_modelkit

        assert hasattr(mada_modelkit, "__all__")

    def test_all_has_17_names(self) -> None:
        from mada_modelkit import __all__

        assert len(__all__) == 17

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
            "TrackingMiddleware",
            "FallbackMiddleware",
        ],
    )
    def test_public_name_importable(self, name: str) -> None:
        import mada_modelkit

        assert hasattr(mada_modelkit, name)
        assert name in mada_modelkit.__all__

    def test_base_client_is_abc(self) -> None:
        from mada_modelkit import BaseAgentClient
        import abc

        assert issubclass(BaseAgentClient, abc.ABC)

    def test_error_hierarchy(self) -> None:
        from mada_modelkit import (
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
        from mada_modelkit import (
            BaseAgentClient,
            ABTestMiddleware,
            RetryMiddleware,
            CircuitBreakerMiddleware,
            CachingMiddleware,
            TrackingMiddleware,
            FallbackMiddleware,
        )

        for cls in [
            ABTestMiddleware,
            RetryMiddleware,
            CircuitBreakerMiddleware,
            CachingMiddleware,
            TrackingMiddleware,
            FallbackMiddleware,
        ]:
            assert issubclass(cls, BaseAgentClient)


class TestProviderImportIsolation:
    """Verify provider subpackages don't fail on import of the module itself."""

    def test_providers_package_imports(self) -> None:
        import mada_modelkit.providers

        assert mada_modelkit.providers is not None

    def test_cloud_package_imports(self) -> None:
        import mada_modelkit.providers.cloud

        assert mada_modelkit.providers.cloud is not None

    def test_local_server_package_imports(self) -> None:
        import mada_modelkit.providers.local_server

        assert mada_modelkit.providers.local_server is not None

    def test_native_package_imports(self) -> None:
        import mada_modelkit.providers.native

        assert mada_modelkit.providers.native is not None
