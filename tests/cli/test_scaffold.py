"""Tests for scaffolding CLI.

Covers provider, middleware, and test template generation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from madakit.cli.scaffold import (
    scaffold_provider,
    scaffold_middleware,
    scaffold_test,
    _to_snake_case,
    _to_pascal_case,
    main,
)


class TestCaseConversion:
    """Test case conversion utilities."""

    def test_to_snake_case_simple(self) -> None:
        """_to_snake_case converts simple names."""
        assert _to_snake_case("MyCustom") == "mycustom"

    def test_to_snake_case_with_hyphen(self) -> None:
        """_to_snake_case converts hyphens to underscores."""
        assert _to_snake_case("my-custom") == "my_custom"

    def test_to_snake_case_with_spaces(self) -> None:
        """_to_snake_case converts spaces to underscores."""
        assert _to_snake_case("my custom") == "my_custom"

    def test_to_pascal_case_simple(self) -> None:
        """_to_pascal_case capitalizes."""
        assert _to_pascal_case("mycustom") == "Mycustom"

    def test_to_pascal_case_with_underscores(self) -> None:
        """_to_pascal_case converts snake_case."""
        assert _to_pascal_case("my_custom") == "MyCustom"

    def test_to_pascal_case_with_hyphens(self) -> None:
        """_to_pascal_case converts kebab-case."""
        assert _to_pascal_case("my-custom") == "MyCustom"


class TestScaffoldProvider:
    """Test provider scaffolding."""

    def test_scaffold_provider_creates_file(self) -> None:
        """scaffold_provider creates file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            filepath = scaffold_provider("MyCustom", output)

            assert filepath.exists()
            assert filepath.name == "mycustom.py"

    def test_scaffold_provider_contains_class(self) -> None:
        """scaffold_provider includes class definition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            filepath = scaffold_provider("MyCustom", output)

            content = filepath.read_text()
            assert "class MyCustomClient" in content
            assert "HttpAgentClient" in content

    def test_scaffold_provider_contains_methods(self) -> None:
        """scaffold_provider includes required methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            filepath = scaffold_provider("TestProvider", output)

            content = filepath.read_text()
            assert "def _build_payload" in content
            assert "def _parse_response" in content
            assert "def _endpoint" in content
            assert "def __repr__" in content

    def test_scaffold_provider_default_output_dir(self) -> None:
        """scaffold_provider uses current directory by default."""
        # Create in temp dir but verify it would use cwd
        import os

        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                filepath = scaffold_provider("Test")

                assert filepath.parent == Path.cwd()
                assert filepath.exists()
        finally:
            os.chdir(original_cwd)


class TestScaffoldMiddleware:
    """Test middleware scaffolding."""

    def test_scaffold_middleware_creates_file(self) -> None:
        """scaffold_middleware creates file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            filepath = scaffold_middleware("MyCustom", output)

            assert filepath.exists()
            assert filepath.name == "mycustom.py"

    def test_scaffold_middleware_contains_class(self) -> None:
        """scaffold_middleware includes class definition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            filepath = scaffold_middleware("MyCustom", output)

            content = filepath.read_text()
            assert "class MyCustomMiddleware" in content
            assert "BaseAgentClient" in content

    def test_scaffold_middleware_contains_methods(self) -> None:
        """scaffold_middleware includes required methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            filepath = scaffold_middleware("TestMiddleware", output)

            content = filepath.read_text()
            assert "async def send_request" in content
            assert "async def send_request_stream" in content
            assert "async def close" in content

    def test_scaffold_middleware_wraps_client(self) -> None:
        """scaffold_middleware includes client wrapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            filepath = scaffold_middleware("Test", output)

            content = filepath.read_text()
            assert "self._client" in content
            assert "await self._client.send_request" in content


class TestScaffoldTest:
    """Test test template scaffolding."""

    def test_scaffold_test_creates_file(self) -> None:
        """scaffold_test creates file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            filepath = scaffold_test("MyCustom", "provider", output)

            assert filepath.exists()
            assert filepath.name == "test_mycustom.py"

    def test_scaffold_test_provider_type(self) -> None:
        """scaffold_test generates provider tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            filepath = scaffold_test("MyCustom", "provider", output)

            content = filepath.read_text()
            assert "MyCustomClient" in content

    def test_scaffold_test_middleware_type(self) -> None:
        """scaffold_test generates middleware tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            filepath = scaffold_test("MyCustom", "middleware", output)

            content = filepath.read_text()
            assert "MyCustomMiddleware" in content

    def test_scaffold_test_contains_test_classes(self) -> None:
        """scaffold_test includes test class structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            filepath = scaffold_test("Test", "provider", output)

            content = filepath.read_text()
            assert "class TestModuleExports" in content
            assert "class TestTestConstructor" in content
            assert "@pytest.mark.asyncio" in content


class TestCLIMain:
    """Test CLI main function."""

    def test_main_provider_command(self, monkeypatch) -> None:
        """main handles provider command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = ["provider", "TestProvider", "-o", tmpdir]
            monkeypatch.setattr("sys.argv", ["scaffold"] + args)

            result = main()

            assert result == 0
            filepath = Path(tmpdir) / "testprovider.py"
            assert filepath.exists()

    def test_main_middleware_command(self, monkeypatch) -> None:
        """main handles middleware command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = ["middleware", "TestMiddleware", "-o", tmpdir]
            monkeypatch.setattr("sys.argv", ["scaffold"] + args)

            result = main()

            assert result == 0
            filepath = Path(tmpdir) / "testmiddleware.py"
            assert filepath.exists()

    def test_main_test_command(self, monkeypatch) -> None:
        """main handles test command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = ["test", "TestComponent", "-t", "provider", "-o", tmpdir]
            monkeypatch.setattr("sys.argv", ["scaffold"] + args)

            result = main()

            assert result == 0
            filepath = Path(tmpdir) / "test_testcomponent.py"
            assert filepath.exists()

    def test_main_no_command_shows_help(self, monkeypatch, capsys) -> None:
        """main without command shows help."""
        monkeypatch.setattr("sys.argv", ["scaffold"])

        result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "usage:" in captured.err.lower()
