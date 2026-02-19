"""Tests for providers/cloud/__init__.py.

Covers: package importability (task 4.5.1) — the cloud sub-package is a
minimal namespace; providers are NOT re-exported from it and must be imported
via their explicit module paths. Verifies that all four provider modules are
reachable by explicit import and that the package does not accidentally expose
provider classes at the top level.
"""

from __future__ import annotations

import importlib
import types


# ---------------------------------------------------------------------------
# TestCloudPackageImport
# ---------------------------------------------------------------------------


class TestCloudPackageImport:
    """Package-level importability for providers/cloud/__init__.py."""

    def test_cloud_package_importable(self) -> None:
        """mada_modelkit.providers.cloud is importable without error."""
        import mada_modelkit.providers.cloud as cloud  # noqa: F401

        assert cloud is not None

    def test_cloud_package_is_module(self) -> None:
        """The cloud package object is a Python module."""
        import mada_modelkit.providers.cloud as cloud

        assert isinstance(cloud, types.ModuleType)

    def test_cloud_package_has_no_all(self) -> None:
        """The cloud package does not define __all__ (no forced re-exports)."""
        import mada_modelkit.providers.cloud as cloud

        assert not hasattr(cloud, "__all__")

    def test_openai_client_not_in_cloud_namespace(self) -> None:
        """OpenAIClient is NOT accessible directly from the cloud package."""
        import mada_modelkit.providers.cloud as cloud

        assert not hasattr(cloud, "OpenAIClient")

    def test_anthropic_client_not_in_cloud_namespace(self) -> None:
        """AnthropicClient is NOT accessible directly from the cloud package."""
        import mada_modelkit.providers.cloud as cloud

        assert not hasattr(cloud, "AnthropicClient")

    def test_gemini_client_not_in_cloud_namespace(self) -> None:
        """GeminiClient is NOT accessible directly from the cloud package."""
        import mada_modelkit.providers.cloud as cloud

        assert not hasattr(cloud, "GeminiClient")

    def test_deepseek_client_not_in_cloud_namespace(self) -> None:
        """DeepSeekClient is NOT accessible directly from the cloud package."""
        import mada_modelkit.providers.cloud as cloud

        assert not hasattr(cloud, "DeepSeekClient")


# ---------------------------------------------------------------------------
# TestExplicitProviderImports
# ---------------------------------------------------------------------------


class TestExplicitProviderImports:
    """Each provider is importable via its explicit module path."""

    def test_openai_module_importable(self) -> None:
        """mada_modelkit.providers.cloud.openai is importable."""
        mod = importlib.import_module("mada_modelkit.providers.cloud.openai")
        assert mod is not None

    def test_anthropic_module_importable(self) -> None:
        """mada_modelkit.providers.cloud.anthropic is importable."""
        mod = importlib.import_module("mada_modelkit.providers.cloud.anthropic")
        assert mod is not None

    def test_gemini_module_importable(self) -> None:
        """mada_modelkit.providers.cloud.gemini is importable."""
        mod = importlib.import_module("mada_modelkit.providers.cloud.gemini")
        assert mod is not None

    def test_deepseek_module_importable(self) -> None:
        """mada_modelkit.providers.cloud.deepseek is importable."""
        mod = importlib.import_module("mada_modelkit.providers.cloud.deepseek")
        assert mod is not None

    def test_openai_client_importable_from_explicit_path(self) -> None:
        """OpenAIClient is importable from mada_modelkit.providers.cloud.openai."""
        from mada_modelkit.providers.cloud.openai import OpenAIClient

        assert OpenAIClient is not None

    def test_anthropic_client_importable_from_explicit_path(self) -> None:
        """AnthropicClient is importable from mada_modelkit.providers.cloud.anthropic."""
        from mada_modelkit.providers.cloud.anthropic import AnthropicClient

        assert AnthropicClient is not None

    def test_gemini_client_importable_from_explicit_path(self) -> None:
        """GeminiClient is importable from mada_modelkit.providers.cloud.gemini."""
        from mada_modelkit.providers.cloud.gemini import GeminiClient

        assert GeminiClient is not None

    def test_deepseek_client_importable_from_explicit_path(self) -> None:
        """DeepSeekClient is importable from mada_modelkit.providers.cloud.deepseek."""
        from mada_modelkit.providers.cloud.deepseek import DeepSeekClient

        assert DeepSeekClient is not None
