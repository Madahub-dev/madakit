"""Prompt template middleware for madakit.

Template management and variable substitution.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import re
from typing import Any, AsyncIterator

from madakit._base import BaseAgentClient
from madakit._errors import MiddlewareError
from madakit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["PromptTemplateMiddleware"]


class PromptTemplateMiddleware(BaseAgentClient):
    """Middleware for prompt template management and variable substitution.

    Supports Jinja2-style {{ variable }} syntax with zero dependencies.
    """

    # Pattern for {{ variable_name }}
    _VARIABLE_PATTERN = re.compile(r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}')

    def __init__(
        self,
        client: BaseAgentClient,
        templates: dict[str, str] | None = None,
    ) -> None:
        """Initialise with client and optional template registry.

        Args:
            client: Wrapped client to delegate to.
            templates: Optional dictionary mapping template names to template strings.
        """
        super().__init__()
        self._client = client
        self._templates = templates or {}

    def register_template(self, name: str, template: str) -> None:
        """Register a new template.

        Args:
            name: Template name for retrieval.
            template: Template string with {{ variable }} placeholders.
        """
        self._templates[name] = template

    def get_template(self, name: str) -> str:
        """Retrieve a template by name.

        Args:
            name: Template name.

        Returns:
            Template string.

        Raises:
            MiddlewareError: If template not found.
        """
        if name not in self._templates:
            raise MiddlewareError(f"Template '{name}' not found")
        return self._templates[name]

    def _extract_variables(self, template: str) -> set[str]:
        """Extract variable names from template.

        Args:
            template: Template string.

        Returns:
            Set of variable names found in template.
        """
        return set(self._VARIABLE_PATTERN.findall(template))

    def _validate_variables(
        self,
        template: str,
        variables: dict[str, Any],
    ) -> None:
        """Validate that all required variables are provided.

        Args:
            template: Template string.
            variables: Variable values.

        Raises:
            MiddlewareError: If required variables are missing.
        """
        required = self._extract_variables(template)
        provided = set(variables.keys())
        missing = required - provided

        if missing:
            missing_list = ", ".join(sorted(missing))
            raise MiddlewareError(f"Missing required variables: {missing_list}")

    def render(self, template: str, variables: dict[str, Any]) -> str:
        """Render a template with variable substitution.

        Args:
            template: Template string with {{ variable }} placeholders.
            variables: Dictionary mapping variable names to values.

        Returns:
            Rendered string with variables substituted.

        Raises:
            MiddlewareError: If required variables are missing.
        """
        # Validate all required variables are provided
        self._validate_variables(template, variables)

        # Substitute variables
        def replace_variable(match: re.Match) -> str:
            var_name = match.group(1)
            value = variables[var_name]
            return str(value)

        return self._VARIABLE_PATTERN.sub(replace_variable, template)

    def render_template(self, name: str, variables: dict[str, Any]) -> str:
        """Render a registered template by name.

        Args:
            name: Template name.
            variables: Variable values.

        Returns:
            Rendered string.

        Raises:
            MiddlewareError: If template not found or variables missing.
        """
        template = self.get_template(name)
        return self.render(template, variables)

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Render prompt template if variables provided, then delegate.

        Args:
            request: The request to send. If metadata contains 'template' or
                    'template_name', the prompt will be treated as a template
                    or the named template will be used. Variables should be in
                    metadata['variables'].

        Returns:
            Response from wrapped client.

        Raises:
            MiddlewareError: If template rendering fails.
        """
        # Check if template rendering is requested
        metadata = request.metadata or {}
        template_name = metadata.get("template_name")
        is_template = metadata.get("template", False)
        variables = metadata.get("variables", {})

        # Determine the template string
        if template_name:
            # Use named template
            template = self.get_template(template_name)
        elif is_template:
            # Treat prompt as template
            template = request.prompt
        else:
            # No templating, pass through
            return await self._client.send_request(request)

        # Render template
        rendered_prompt = self.render(template, variables)

        # Create new request with rendered prompt
        rendered_request = AgentRequest(
            prompt=rendered_prompt,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
            attachments=request.attachments,
            metadata=request.metadata,
        )

        # Delegate to wrapped client
        return await self._client.send_request(rendered_request)

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Render prompt template if variables provided, then stream.

        Args:
            request: The request to send. Template rendering works same as send_request.

        Yields:
            Stream chunks from wrapped client.

        Raises:
            MiddlewareError: If template rendering fails.
        """
        # Check if template rendering is requested
        metadata = request.metadata or {}
        template_name = metadata.get("template_name")
        is_template = metadata.get("template", False)
        variables = metadata.get("variables", {})

        # Determine the template string
        if template_name:
            # Use named template
            template = self.get_template(template_name)
        elif is_template:
            # Treat prompt as template
            template = request.prompt
        else:
            # No templating, pass through
            async for chunk in self._client.send_request_stream(request):
                yield chunk
            return

        # Render template
        rendered_prompt = self.render(template, variables)

        # Create new request with rendered prompt
        rendered_request = AgentRequest(
            prompt=rendered_prompt,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
            attachments=request.attachments,
            metadata=request.metadata,
        )

        # Stream from wrapped client
        async for chunk in self._client.send_request_stream(rendered_request):
            yield chunk
