"""Migration tools for mada-modelkit.

Convert code and configs from other frameworks to mada-modelkit.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

__all__ = ["migrate_langchain", "convert_config", "check_compatibility", "main"]


def migrate_langchain(source_code: str) -> str:
    """Convert LangChain code to mada-modelkit.

    Args:
        source_code: Python source code using LangChain.

    Returns:
        Converted code using mada-modelkit.

    Example:
        >>> code = "from langchain.llms import OpenAI"
        >>> result = migrate_langchain(code)
        >>> "from madakit" in result
        True
    """
    result = source_code

    # Replace imports
    result = re.sub(
        r"from langchain\.llms import (\w+)",
        r"from madakit.providers.cloud.\1 import \1Client",
        result,
    )
    result = re.sub(
        r"from langchain import (\w+)",
        r"# TODO: Manual migration needed for: \1",
        result,
    )

    # Replace LLM instantiation
    result = re.sub(
        r"(\w+)\s*=\s*OpenAI\(",
        r"\1 = OpenAIClient(",
        result,
    )

    # Replace common LangChain patterns
    result = re.sub(
        r"\.predict\((.*?)\)",
        r".send_request(AgentRequest(prompt=\1))",
        result,
    )
    result = re.sub(
        r"LLMChain\((.*?)\)",
        r"# TODO: Replace LLMChain with direct client usage",
        result,
    )

    # Add import for AgentRequest if needed
    if "AgentRequest" in result and "from madakit" not in result:
        result = "from madakit._types import AgentRequest\n" + result

    return result


def convert_config(config_data: dict[str, Any], from_format: str = "langchain") -> dict[str, Any]:
    """Convert configuration from other frameworks.

    Args:
        config_data: Configuration dict from other framework.
        from_format: Source format ("langchain", "llamaindex").

    Returns:
        Converted configuration for mada-modelkit.

    Example:
        >>> config = {"model_name": "gpt-3.5-turbo", "temperature": 0.7}
        >>> result = convert_config(config)
        >>> result["model"]
        'gpt-3.5-turbo'
    """
    if from_format == "langchain":
        return _convert_langchain_config(config_data)
    else:
        raise ValueError(f"Unsupported format: {from_format}")


def _convert_langchain_config(config: dict[str, Any]) -> dict[str, Any]:
    """Convert LangChain config to mada-modelkit format.

    Args:
        config: LangChain configuration.

    Returns:
        Mada-modelkit configuration.
    """
    result: dict[str, Any] = {
        "provider": {
            "type": "openai",  # Default assumption
            "model": config.get("model_name", "gpt-3.5-turbo"),
        }
    }

    # Map common parameters
    if "temperature" in config:
        result["provider"]["temperature"] = config["temperature"]

    if "max_tokens" in config:
        result["provider"]["max_tokens"] = config["max_tokens"]

    if "api_key" in config:
        result["provider"]["api_key"] = config["api_key"]

    # Convert middleware-like features
    middleware = []

    if config.get("cache"):
        middleware.append({"type": "cache", "params": {"ttl_seconds": 3600}})

    if config.get("max_retries", 0) > 0:
        middleware.append(
            {
                "type": "retry",
                "params": {
                    "max_retries": config["max_retries"],
                    "backoff_factor": 2.0,
                },
            }
        )

    if middleware:
        result["middleware"] = middleware

    return result


def check_compatibility(code: str) -> tuple[bool, list[str]]:
    """Check if code can be migrated to mada-modelkit.

    Args:
        code: Python source code to check.

    Returns:
        Tuple of (is_compatible, list of warnings/issues).

    Example:
        >>> code = "from langchain.llms import OpenAI"
        >>> compatible, issues = check_compatibility(code)
        >>> compatible
        True
    """
    issues = []

    # Check for unsupported features
    unsupported_patterns = [
        (r"from langchain\.agents import", "LangChain agents - requires manual migration"),
        (r"from langchain\.memory import", "LangChain memory - use custom implementation"),
        (r"ConversationChain", "ConversationChain - use custom conversation logic"),
        (r"\.run\(", "Chain.run() - replace with send_request()"),
    ]

    for pattern, message in unsupported_patterns:
        if re.search(pattern, code):
            issues.append(f"Warning: {message}")

    # Check for compatible patterns
    compatible_patterns = [
        r"from langchain\.llms import",
        r"OpenAI\(",
        r"\.predict\(",
    ]

    has_compatible = any(re.search(p, code) for p in compatible_patterns)

    is_compatible = has_compatible and len(issues) < 3

    if not has_compatible:
        issues.append("No recognized LangChain patterns found")

    return is_compatible, issues


def main() -> int:
    """CLI entry point for migration tools.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Migrate code and configs to mada-modelkit"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # LangChain migration
    langchain_parser = subparsers.add_parser(
        "langchain", help="Migrate LangChain code"
    )
    langchain_parser.add_argument("input", type=Path, help="Input file")
    langchain_parser.add_argument(
        "-o", "--output", type=Path, help="Output file (default: input + .migrated)"
    )

    # Compatibility check
    check_parser = subparsers.add_parser(
        "check", help="Check migration compatibility"
    )
    check_parser.add_argument("input", type=Path, help="Input file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "langchain":
            source = args.input.read_text()
            migrated = migrate_langchain(source)

            output_path = args.output or Path(f"{args.input}.migrated")
            output_path.write_text(migrated)
            print(f"Migrated code written to: {output_path}")

        elif args.command == "check":
            source = args.input.read_text()
            compatible, issues = check_compatibility(source)

            if compatible:
                print("✓ Code appears compatible with migration")
            else:
                print("✗ Code may have compatibility issues")

            if issues:
                print("\nIssues found:")
                for issue in issues:
                    print(f"  - {issue}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
