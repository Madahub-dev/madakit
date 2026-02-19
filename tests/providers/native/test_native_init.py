"""Tests for providers/native/__init__.py.

Covers: package importability; absence of __all__ and provider re-exports at the
package level (providers are imported via explicit module paths, not from the
package root); both LlamaCppClient and TransformersClient reachable via their
explicit module paths.
"""

from __future__ import annotations

import importlib
import types


# ---------------------------------------------------------------------------
# TestNativePackageImport
# ---------------------------------------------------------------------------


class TestNativePackageImport:
    """providers/native/__init__.py package-level contract (task 6.3.1)."""

    def test_native_package_importable(self) -> None:
        """The providers.native package is importable."""
        import mada_modelkit.providers.native as native  # noqa: F401

        assert native is not None

    def test_native_package_is_a_package(self) -> None:
        """providers.native is a package (has __path__)."""
        import mada_modelkit.providers.native as native

        assert hasattr(native, "__path__")

    def test_native_package_has_no_all(self) -> None:
        """__all__ is not defined in providers/native/__init__.py."""
        import mada_modelkit.providers.native as native

        assert not hasattr(native, "__all__")

    def test_llamacpp_client_not_in_native_namespace(self) -> None:
        """LlamaCppClient is NOT exported from the native package namespace."""
        import mada_modelkit.providers.native as native

        assert not hasattr(native, "LlamaCppClient")

    def test_transformers_client_not_in_native_namespace(self) -> None:
        """TransformersClient is NOT exported from the native package namespace."""
        import mada_modelkit.providers.native as native

        assert not hasattr(native, "TransformersClient")

    def test_native_package_has_docstring(self) -> None:
        """providers/native/__init__.py has a module-level docstring."""
        import mada_modelkit.providers.native as native

        assert native.__doc__ is not None
        assert len(native.__doc__.strip()) > 0

    def test_native_package_is_module(self) -> None:
        """providers.native resolves to a types.ModuleType instance."""
        import mada_modelkit.providers.native as native

        assert isinstance(native, types.ModuleType)

    def test_native_package_importable_via_importlib(self) -> None:
        """providers.native is importable via importlib.import_module."""
        native = importlib.import_module("mada_modelkit.providers.native")
        assert native is not None


# ---------------------------------------------------------------------------
# TestExplicitProviderImports
# ---------------------------------------------------------------------------


class TestExplicitProviderImports:
    """Both native providers are reachable via their explicit module paths."""

    def test_llamacpp_module_importable(self) -> None:
        """mada_modelkit.providers.native.llamacpp is importable."""
        import mada_modelkit.providers.native.llamacpp as mod  # noqa: F401

        assert mod is not None

    def test_llamacpp_client_importable_from_module(self) -> None:
        """LlamaCppClient is importable from its explicit module path."""
        from mada_modelkit.providers.native.llamacpp import LlamaCppClient

        assert LlamaCppClient is not None

    def test_transformers_module_importable(self) -> None:
        """mada_modelkit.providers.native.transformers is importable."""
        import mada_modelkit.providers.native.transformers as mod  # noqa: F401

        assert mod is not None

    def test_transformers_client_importable_from_module(self) -> None:
        """TransformersClient is importable from its explicit module path."""
        from mada_modelkit.providers.native.transformers import TransformersClient

        assert TransformersClient is not None
