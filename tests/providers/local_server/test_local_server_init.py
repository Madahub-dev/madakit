"""Tests for providers/local_server/__init__.py.

Covers: package importability (task 5.4.1) — the local_server sub-package is a
minimal namespace; providers are NOT re-exported from it and must be imported
via their explicit module paths. Verifies that all three provider modules are
reachable by explicit import and that the package does not accidentally expose
provider classes at the top level.
"""

from __future__ import annotations

import importlib
import types


# ---------------------------------------------------------------------------
# TestLocalServerPackageImport
# ---------------------------------------------------------------------------


class TestLocalServerPackageImport:
    """Package-level importability for providers/local_server/__init__.py."""

    def test_local_server_package_importable(self) -> None:
        """mada_modelkit.providers.local_server is importable without error."""
        import mada_modelkit.providers.local_server as local_server  # noqa: F401

        assert local_server is not None

    def test_local_server_package_is_module(self) -> None:
        """The local_server package object is a Python module."""
        import mada_modelkit.providers.local_server as local_server

        assert isinstance(local_server, types.ModuleType)

    def test_local_server_package_has_no_all(self) -> None:
        """The local_server package does not define __all__ (no forced re-exports)."""
        import mada_modelkit.providers.local_server as local_server

        assert not hasattr(local_server, "__all__")

    def test_ollama_client_not_in_local_server_namespace(self) -> None:
        """OllamaClient is NOT accessible directly from the local_server package."""
        import mada_modelkit.providers.local_server as local_server

        assert not hasattr(local_server, "OllamaClient")

    def test_vllm_client_not_in_local_server_namespace(self) -> None:
        """VllmClient is NOT accessible directly from the local_server package."""
        import mada_modelkit.providers.local_server as local_server

        assert not hasattr(local_server, "VllmClient")

    def test_localai_client_not_in_local_server_namespace(self) -> None:
        """LocalAIClient is NOT accessible directly from the local_server package."""
        import mada_modelkit.providers.local_server as local_server

        assert not hasattr(local_server, "LocalAIClient")


# ---------------------------------------------------------------------------
# TestExplicitProviderImports
# ---------------------------------------------------------------------------


class TestExplicitProviderImports:
    """Each provider is importable via its explicit module path."""

    def test_ollama_module_importable(self) -> None:
        """mada_modelkit.providers.local_server.ollama is importable."""
        mod = importlib.import_module("mada_modelkit.providers.local_server.ollama")
        assert mod is not None

    def test_vllm_module_importable(self) -> None:
        """mada_modelkit.providers.local_server.vllm is importable."""
        mod = importlib.import_module("mada_modelkit.providers.local_server.vllm")
        assert mod is not None

    def test_localai_module_importable(self) -> None:
        """mada_modelkit.providers.local_server.localai is importable."""
        mod = importlib.import_module("mada_modelkit.providers.local_server.localai")
        assert mod is not None

    def test_ollama_client_importable_from_explicit_path(self) -> None:
        """OllamaClient is importable from mada_modelkit.providers.local_server.ollama."""
        from mada_modelkit.providers.local_server.ollama import OllamaClient

        assert OllamaClient is not None

    def test_vllm_client_importable_from_explicit_path(self) -> None:
        """VllmClient is importable from mada_modelkit.providers.local_server.vllm."""
        from mada_modelkit.providers.local_server.vllm import VllmClient

        assert VllmClient is not None

    def test_localai_client_importable_from_explicit_path(self) -> None:
        """LocalAIClient is importable from mada_modelkit.providers.local_server.localai."""
        from mada_modelkit.providers.local_server.localai import LocalAIClient

        assert LocalAIClient is not None
