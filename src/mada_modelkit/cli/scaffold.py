"""Scaffolding CLI for mada-modelkit.

Generates boilerplate code for custom providers, middleware, and tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["scaffold_provider", "scaffold_middleware", "scaffold_test", "main"]


def _to_snake_case(name: str) -> str:
    """Convert name to snake_case.

    Args:
        name: Name to convert.

    Returns:
        snake_case version of name.
    """
    # Simple conversion: lowercase and replace spaces/hyphens with underscores
    return name.lower().replace("-", "_").replace(" ", "_")


def _to_pascal_case(name: str) -> str:
    """Convert name to PascalCase.

    Args:
        name: Name to convert.

    Returns:
        PascalCase version of name.
    """
    # If already in PascalCase (contains uppercase but no separators), return as-is
    if "_" not in name and "-" not in name and " " not in name:
        # Ensure first letter is capitalized
        return name[0].upper() + name[1:] if name else name

    # Split on underscores, hyphens, and spaces, then capitalize each part
    parts = name.replace("-", "_").replace(" ", "_").split("_")
    return "".join(word.capitalize() for word in parts if word)


def scaffold_provider(name: str, output_dir: Path | None = None) -> Path:
    """Generate custom provider boilerplate.

    Args:
        name: Provider name (e.g., "MyCustom").
        output_dir: Output directory (default: current directory).

    Returns:
        Path to generated file.

    Example:
        >>> scaffold_provider("MyCustom")
        PosixPath('my_custom.py')
    """
    if output_dir is None:
        output_dir = Path.cwd()

    snake_name = _to_snake_case(name)
    pascal_name = _to_pascal_case(name)
    filename = f"{snake_name}.py"
    filepath = output_dir / filename

    template = f'''"""Custom {pascal_name} provider for mada-modelkit.

{pascal_name} provider implementation.
"""

from __future__ import annotations

from typing import Any

from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.providers._http_base import HttpAgentClient

__all__ = ["{pascal_name}Client"]


class {pascal_name}Client(HttpAgentClient):
    """Client for {pascal_name} API.

    Custom provider implementation for {pascal_name}.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "default-model",
        base_url: str = "https://api.example.com/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize {pascal_name} client.

        Args:
            api_key: {pascal_name} API key.
            model: Model name (default: default-model).
            base_url: API base URL.
            **kwargs: Additional arguments for HttpAgentClient.
        """
        super().__init__(
            base_url=base_url,
            headers={{
                "Authorization": f"Bearer {{api_key}}",
            }},
            **kwargs,
        )
        self._model = model
        self._api_key = api_key

    def __repr__(self) -> str:
        """Return string representation with redacted API key."""
        return f"{pascal_name}Client(model={{self._model!r}}, api_key='***')"

    def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
        """Build {pascal_name} request payload.

        Args:
            request: The request to convert.

        Returns:
            {pascal_name} API payload dict.
        """
        payload: dict[str, Any] = {{
            "model": self._model,
            "prompt": request.prompt,
        }}

        # Add optional parameters
        if request.max_tokens != 1024:
            payload["max_tokens"] = request.max_tokens

        if request.temperature != 0.7:
            payload["temperature"] = request.temperature

        if request.stop:
            payload["stop"] = request.stop

        if request.system_prompt:
            payload["system_prompt"] = request.system_prompt

        return payload

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse {pascal_name} API response.

        Args:
            data: Response JSON from {pascal_name} API.

        Returns:
            Parsed AgentResponse.
        """
        # TODO: Adjust fields based on actual API response format
        content = data.get("text", data.get("content", ""))
        model = data.get("model", self._model)
        input_tokens = data.get("usage", {{}}).get("prompt_tokens", 0)
        output_tokens = data.get("usage", {{}}).get("completion_tokens", 0)

        return AgentResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _endpoint(self) -> str:
        """Return {pascal_name} API endpoint.

        Returns:
            API endpoint path.
        """
        # TODO: Adjust endpoint based on actual API
        return "/completions"
'''

    filepath.write_text(template)
    return filepath


def scaffold_middleware(name: str, output_dir: Path | None = None) -> Path:
    """Generate custom middleware boilerplate.

    Args:
        name: Middleware name (e.g., "MyCustom").
        output_dir: Output directory (default: current directory).

    Returns:
        Path to generated file.

    Example:
        >>> scaffold_middleware("MyCustom")
        PosixPath('my_custom.py')
    """
    if output_dir is None:
        output_dir = Path.cwd()

    snake_name = _to_snake_case(name)
    pascal_name = _to_pascal_case(name)
    filename = f"{snake_name}.py"
    filepath = output_dir / filename

    template = f'''"""Custom {pascal_name} middleware for mada-modelkit.

{pascal_name} middleware implementation.
"""

from __future__ import annotations

from typing import AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["{pascal_name}Middleware"]


class {pascal_name}Middleware(BaseAgentClient):
    """Middleware for {pascal_name.lower()} functionality.

    Wraps another client and adds {pascal_name.lower()} behavior.
    """

    def __init__(
        self,
        client: BaseAgentClient,
        # TODO: Add custom parameters
        **kwargs,
    ) -> None:
        """Initialize {pascal_name} middleware.

        Args:
            client: The wrapped client.
            **kwargs: Additional arguments for BaseAgentClient.
        """
        super().__init__(**kwargs)
        self._client = client
        # TODO: Initialize custom state

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Send request with {pascal_name.lower()} processing.

        Args:
            request: The request to send.

        Returns:
            Response from wrapped client.
        """
        # TODO: Add pre-processing logic here

        # Delegate to wrapped client
        response = await self._client.send_request(request)

        # TODO: Add post-processing logic here

        return response

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming request with {pascal_name.lower()} processing.

        Args:
            request: The request to send.

        Yields:
            Stream chunks from wrapped client.
        """
        # TODO: Add pre-processing logic here

        # Stream from wrapped client
        async for chunk in self._client.send_request_stream(request):
            # TODO: Add per-chunk processing logic here
            yield chunk

        # TODO: Add post-stream logic here

    async def close(self) -> None:
        """Close middleware and wrapped client."""
        await self._client.close()
'''

    filepath.write_text(template)
    return filepath


def scaffold_test(name: str, target_type: str = "provider", output_dir: Path | None = None) -> Path:
    """Generate test file template.

    Args:
        name: Name of the component being tested.
        target_type: Type of component ("provider" or "middleware").
        output_dir: Output directory (default: current directory).

    Returns:
        Path to generated file.

    Example:
        >>> scaffold_test("MyCustom", "provider")
        PosixPath('test_my_custom.py')
    """
    if output_dir is None:
        output_dir = Path.cwd()

    snake_name = _to_snake_case(name)
    pascal_name = _to_pascal_case(name)
    filename = f"test_{snake_name}.py"
    filepath = output_dir / filename

    if target_type == "provider":
        class_name = f"{pascal_name}Client"
        module_path = snake_name
    else:
        class_name = f"{pascal_name}Middleware"
        module_path = snake_name

    template = f'''"""Tests for {pascal_name}.

Covers {class_name} functionality.
"""

from __future__ import annotations

import pytest

from mada_modelkit._types import AgentRequest
# TODO: Adjust import path
from {module_path} import {class_name}


class TestModuleExports:
    """Verify module exports."""

    def test_module_has_all(self) -> None:
        """Module exports __all__."""
        import {module_path}

        assert hasattr({module_path}, "__all__")

    def test_all_contains_class(self) -> None:
        """__all__ contains {class_name}."""
        from {module_path} import __all__

        assert "{class_name}" in __all__


class Test{pascal_name}Constructor:
    """Test {class_name} constructor."""

    def test_valid_constructor(self) -> None:
        """Constructor initializes correctly."""
        # TODO: Adjust constructor arguments
        client = {class_name}(api_key="test-key")

        assert client is not None

    def test_custom_parameters(self) -> None:
        """Constructor accepts custom parameters."""
        # TODO: Test custom parameter handling
        pass


class Test{pascal_name}SendRequest:
    """Test send_request method."""

    @pytest.mark.asyncio
    async def test_send_request(self) -> None:
        """send_request returns response."""
        # TODO: Implement test with mocked HTTP or wrapped client
        pass


class Test{pascal_name}Integration:
    """Integration tests for {class_name}."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Client works as async context manager."""
        # TODO: Test async context manager
        pass
'''

    filepath.write_text(template)
    return filepath


def main() -> int:
    """CLI entry point for scaffolding.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Scaffold boilerplate code for mada-modelkit"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Provider scaffolding
    provider_parser = subparsers.add_parser(
        "provider", help="Generate provider boilerplate"
    )
    provider_parser.add_argument("name", help="Provider name (e.g., MyCustom)")
    provider_parser.add_argument(
        "-o", "--output", type=Path, help="Output directory (default: current)"
    )

    # Middleware scaffolding
    middleware_parser = subparsers.add_parser(
        "middleware", help="Generate middleware boilerplate"
    )
    middleware_parser.add_argument("name", help="Middleware name (e.g., MyCustom)")
    middleware_parser.add_argument(
        "-o", "--output", type=Path, help="Output directory (default: current)"
    )

    # Test scaffolding
    test_parser = subparsers.add_parser("test", help="Generate test template")
    test_parser.add_argument("name", help="Component name (e.g., MyCustom)")
    test_parser.add_argument(
        "-t",
        "--type",
        choices=["provider", "middleware"],
        default="provider",
        help="Component type",
    )
    test_parser.add_argument(
        "-o", "--output", type=Path, help="Output directory (default: current)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "provider":
            filepath = scaffold_provider(args.name, args.output)
            print(f"Generated provider: {filepath}")
        elif args.command == "middleware":
            filepath = scaffold_middleware(args.name, args.output)
            print(f"Generated middleware: {filepath}")
        elif args.command == "test":
            filepath = scaffold_test(args.name, args.type, args.output)
            print(f"Generated test: {filepath}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
