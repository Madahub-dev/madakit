"""Tests for migration tools.

Covers LangChain migration, config conversion, and compatibility checking.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from madakit.cli.migrate import (
    migrate_langchain,
    convert_config,
    check_compatibility,
    main,
)


class TestMigrateLangChain:
    """Test LangChain code migration."""

    def test_migrate_simple_import(self) -> None:
        """migrate_langchain converts import."""
        code = "from langchain.llms import OpenAI"

        result = migrate_langchain(code)

        assert "from madakit.providers.cloud.OpenAI import OpenAIClient" in result

    def test_migrate_instantiation(self) -> None:
        """migrate_langchain converts instantiation."""
        code = "llm = OpenAI(temperature=0.7)"

        result = migrate_langchain(code)

        assert "llm = OpenAIClient(temperature=0.7)" in result

    def test_migrate_predict_call(self) -> None:
        """migrate_langchain converts predict calls."""
        code = 'result = llm.predict("Hello")'

        result = migrate_langchain(code)

        assert "send_request" in result
        assert "AgentRequest" in result

    def test_migrate_adds_agent_request_import(self) -> None:
        """migrate_langchain adds AgentRequest import when needed."""
        code = 'result = llm.predict("Hello")'

        result = migrate_langchain(code)

        assert "from madakit._types import AgentRequest" in result

    def test_migrate_llmchain_comment(self) -> None:
        """migrate_langchain adds TODO for LLMChain."""
        code = "chain = LLMChain(llm=llm)"

        result = migrate_langchain(code)

        assert "TODO" in result


class TestConvertConfig:
    """Test configuration conversion."""

    def test_convert_basic_config(self) -> None:
        """convert_config converts basic parameters."""
        config = {
            "model_name": "gpt-4",
            "temperature": 0.9,
            "max_tokens": 500,
        }

        result = convert_config(config, "langchain")

        assert result["provider"]["model"] == "gpt-4"
        assert result["provider"]["temperature"] == 0.9
        assert result["provider"]["max_tokens"] == 500

    def test_convert_with_cache(self) -> None:
        """convert_config adds cache middleware."""
        config = {
            "model_name": "gpt-3.5-turbo",
            "cache": True,
        }

        result = convert_config(config, "langchain")

        assert "middleware" in result
        assert any(m["type"] == "cache" for m in result["middleware"])

    def test_convert_with_retries(self) -> None:
        """convert_config adds retry middleware."""
        config = {
            "model_name": "gpt-3.5-turbo",
            "max_retries": 3,
        }

        result = convert_config(config, "langchain")

        assert "middleware" in result
        retry_mw = next(m for m in result["middleware"] if m["type"] == "retry")
        assert retry_mw["params"]["max_retries"] == 3

    def test_convert_unsupported_format(self) -> None:
        """convert_config raises on unsupported format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            convert_config({}, "unknown")

    def test_convert_defaults_model(self) -> None:
        """convert_config provides default model."""
        config = {}

        result = convert_config(config, "langchain")

        assert result["provider"]["model"] == "gpt-3.5-turbo"


class TestCheckCompatibility:
    """Test compatibility checking."""

    def test_check_compatible_code(self) -> None:
        """check_compatibility returns True for compatible code."""
        code = """
        from langchain.llms import OpenAI
        llm = OpenAI()
        result = llm.predict("Hello")
        """

        compatible, issues = check_compatibility(code)

        assert compatible is True

    def test_check_incompatible_agents(self) -> None:
        """check_compatibility warns about agents."""
        code = "from langchain.agents import AgentExecutor"

        compatible, issues = check_compatibility(code)

        assert any("agents" in issue.lower() for issue in issues)

    def test_check_incompatible_memory(self) -> None:
        """check_compatibility warns about memory."""
        code = "from langchain.memory import ConversationBufferMemory"

        compatible, issues = check_compatibility(code)

        assert any("memory" in issue.lower() for issue in issues)

    def test_check_no_patterns(self) -> None:
        """check_compatibility detects no recognizable patterns."""
        code = "print('Hello world')"

        compatible, issues = check_compatibility(code)

        assert compatible is False
        assert any("No recognized" in issue for issue in issues)

    def test_check_returns_issues_list(self) -> None:
        """check_compatibility returns list of issues."""
        code = """
        from langchain.agents import AgentExecutor
        from langchain.memory import ConversationBufferMemory
        """

        compatible, issues = check_compatibility(code)

        assert len(issues) >= 2


class TestCLIMain:
    """Test CLI main function."""

    def test_main_langchain_command(self, monkeypatch) -> None:
        """main handles langchain migration command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "test.py"
            input_file.write_text("from langchain.llms import OpenAI")

            args = ["langchain", str(input_file)]
            monkeypatch.setattr("sys.argv", ["migrate"] + args)

            result = main()

            assert result == 0
            output_file = Path(f"{input_file}.migrated")
            assert output_file.exists()

    def test_main_check_command(self, monkeypatch, capsys) -> None:
        """main handles check command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "test.py"
            input_file.write_text("from langchain.llms import OpenAI")

            args = ["check", str(input_file)]
            monkeypatch.setattr("sys.argv", ["migrate"] + args)

            result = main()

            assert result == 0
            captured = capsys.readouterr()
            assert "compatible" in captured.out.lower()

    def test_main_no_command_shows_help(self, monkeypatch, capsys) -> None:
        """main without command shows help."""
        monkeypatch.setattr("sys.argv", ["migrate"])

        result = main()

        assert result == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "usage:" in captured.err.lower()
