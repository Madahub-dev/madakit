# Extension Guide

Learn how to build custom providers and middleware for **madakit**.

---

## Table of Contents

1. [Building Custom Providers](#building-custom-providers)
2. [Building Custom Middleware](#building-custom-middleware)
3. [Testing Your Extensions](#testing-your-extensions)
4. [Publishing Extensions](#publishing-extensions)

---

## Building Custom Providers

### Provider Checklist

Every provider must:
- [x] Inherit from `BaseAgentClient`
- [x] Implement `send_request(AgentRequest) -> AgentResponse`
- [x] Optionally override `send_request_stream()` for streaming
- [x] Implement `close()` if resources need cleanup
- [x] Implement `health_check()` if applicable
- [x] Redact API keys in `__repr__`

---

### Example 1: Custom HTTP Provider

```python
from madakit.providers._http_base import HttpAgentClient
from madakit import AgentRequest, AgentResponse

class CustomCloudClient(HttpAgentClient):
    """Client for a custom cloud API."""

    _require_tls: bool = True  # Enforce HTTPS

    def __init__(
        self,
        api_key: str,
        model: str = "custom-model-v1",
        base_url: str = "https://api.example.com/v1",
        **kwargs
    ):
        super().__init__(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            **kwargs
        )
        self._api_key = api_key
        self._model = model

    def __repr__(self) -> str:
        """Redact API key in repr."""
        return f"CustomCloudClient(model={self._model!r}, api_key='***')"

    def _build_payload(self, request: AgentRequest) -> dict:
        """Convert AgentRequest to provider-specific JSON."""
        payload = {
            "model": self._model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt

        if request.stop:
            payload["stop"] = request.stop

        return payload

    def _parse_response(self, data: dict) -> AgentResponse:
        """Convert provider JSON to AgentResponse."""
        return AgentResponse(
            content=data["output"]["text"],
            model=data.get("model", self._model),
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
        )

    def _endpoint(self) -> str:
        """Return API endpoint path."""
        return "/completions"
```

**Usage:**
```python
client = CustomCloudClient(api_key="sk-...")
response = await client.send_request(request)
```

---

### Example 2: Custom Streaming Provider

```python
from typing import AsyncIterator
from madakit import StreamChunk

class CustomStreamingClient(HttpAgentClient):
    """Provider with SSE streaming support."""

    _require_tls: bool = True

    # ... __init__, _build_payload, _parse_response, _endpoint same as above

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Stream response chunks via SSE."""
        payload = self._build_payload(request)
        payload["stream"] = True

        try:
            async with self._client.stream(
                "POST",
                self._build_url(self._endpoint()),
                json=payload,
                headers=self._headers,
                timeout=self._timeout,
            ) as response:
                self._check_status(response)

                # Parse server-sent events
                async for line in response.aiter_lines():
                    if not line or line.startswith("event:"):
                        continue

                    # Remove "data: " prefix
                    if line.startswith("data: "):
                        line = line[6:]

                    if line.strip() == "[DONE]":
                        break

                    try:
                        chunk_data = self._json.loads(line)

                        # Extract delta
                        delta = chunk_data.get("delta", "")
                        is_final = chunk_data.get("finish_reason") is not None

                        if delta or is_final:
                            metadata = {}
                            if is_final:
                                metadata.update({
                                    "model": chunk_data.get("model"),
                                    "input_tokens": chunk_data.get("usage", {}).get("input_tokens", 0),
                                    "output_tokens": chunk_data.get("usage", {}).get("output_tokens", 0),
                                })

                            yield StreamChunk(
                                delta=delta,
                                is_final=is_final,
                                metadata=metadata
                            )

                    except self._json.JSONDecodeError:
                        continue  # Skip malformed chunks

        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Streaming request failed: {e}") from e
```

---

### Example 3: Custom Native Provider (In-Process)

```python
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio

class CustomNativeClient(BaseAgentClient):
    """In-process provider using a Python library."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self._model_path = model_path
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __repr__(self) -> str:
        return f"CustomNativeClient(model_path={self._model_path!r})"

    async def __aenter__(self):
        """Load model asynchronously (non-blocking)."""
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            self._executor,
            self._load_model
        )
        return self

    def _load_model(self):
        """Load model in thread pool (blocking operation)."""
        # Import library lazily (optional dependency)
        from some_library import Model
        return Model.load(self._model_path)

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute inference in thread pool."""
        if self._model is None:
            await self.__aenter__()  # Lazy load

        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self._executor,
                partial(self._sync_generate, request)
            )
        except Exception as exc:
            raise ProviderError(f"Inference failed: {exc}") from exc

    def _sync_generate(self, request: AgentRequest) -> AgentResponse:
        """Synchronous inference (runs in thread pool)."""
        output = self._model.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        return AgentResponse(
            content=output["text"],
            model=self._model_path,
            input_tokens=output.get("input_tokens", 0),
            output_tokens=output.get("output_tokens", 0)
        )

    async def close(self) -> None:
        """Cleanup resources."""
        self._executor.shutdown(wait=False)
        self._model = None
```

---

### Example 4: Using the Scaffolding CLI

```bash
# Generate provider boilerplate
python -m madakit.cli.scaffold provider MyCustomProvider --output my_provider.py

# Generate test template
python -m madakit.cli.scaffold test MyCustomProvider --output test_my_provider.py
```

**Generated `my_provider.py`:**
```python
from madakit.providers._http_base import HttpAgentClient
from madakit import AgentRequest, AgentResponse

class MyCustomProvider(HttpAgentClient):
    """Custom provider implementation."""

    _require_tls: bool = True

    def __init__(self, api_key: str, model: str = "default", **kwargs):
        super().__init__(
            base_url="https://api.example.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            **kwargs
        )
        self._api_key = api_key
        self._model = model

    def __repr__(self) -> str:
        return f"MyCustomProvider(model={self._model!r}, api_key='***')"

    def _build_payload(self, request: AgentRequest) -> dict:
        # TODO: Implement payload construction
        return {
            "model": self._model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

    def _parse_response(self, data: dict) -> AgentResponse:
        # TODO: Implement response parsing
        return AgentResponse(
            content=data["text"],
            model=self._model,
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
        )

    def _endpoint(self) -> str:
        # TODO: Return API endpoint
        return "/completions"
```

---

## Building Custom Middleware

### Middleware Checklist

Every middleware must:
- [x] Inherit from `BaseAgentClient`
- [x] Wrap another `BaseAgentClient` in `__init__`
- [x] Implement **both** `send_request` and `send_request_stream`
- [x] Delegate to wrapped client (don't break the chain)
- [x] Preserve all request/response fields
- [x] Optionally implement `close()` to delegate cleanup

---

### Example 1: Simple Logging Middleware

```python
import time
import logging
from typing import AsyncIterator

class SimpleLoggingMiddleware(BaseAgentClient):
    """Middleware that logs requests and responses."""

    def __init__(self, client: BaseAgentClient, logger: logging.Logger | None = None):
        super().__init__()
        self._client = client
        self._logger = logger or logging.getLogger(__name__)

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Log request and response."""
        start_time = time.perf_counter()

        self._logger.info(
            f"Request: prompt={request.prompt[:50]}..., "
            f"max_tokens={request.max_tokens}, temp={request.temperature}"
        )

        try:
            response = await self._client.send_request(request)

            duration_ms = (time.perf_counter() - start_time) * 1000
            self._logger.info(
                f"Response: model={response.model}, tokens={response.total_tokens}, "
                f"duration={duration_ms:.0f}ms"
            )

            return response

        except Exception as e:
            self._logger.error(f"Error: {type(e).__name__}: {e}")
            raise

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Log streaming request."""
        self._logger.info(f"Streaming request: {request.prompt[:50]}...")

        chunk_count = 0
        async for chunk in self._client.send_request_stream(request):
            chunk_count += 1
            yield chunk

            if chunk.is_final:
                self._logger.info(f"Stream complete: {chunk_count} chunks")
```

**Usage:**
```python
client = SimpleLoggingMiddleware(OpenAIClient(api_key="..."))
response = await client.send_request(request)
```

---

### Example 2: Request Transformation Middleware

```python
class PromptPrefixMiddleware(BaseAgentClient):
    """Middleware that adds a prefix to all prompts."""

    def __init__(self, client: BaseAgentClient, prefix: str):
        super().__init__()
        self._client = client
        self._prefix = prefix

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Add prefix to prompt."""
        modified_request = AgentRequest(
            prompt=f"{self._prefix}\n\n{request.prompt}",
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
            attachments=request.attachments,
            metadata=request.metadata,
        )

        return await self._client.send_request(modified_request)

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Add prefix to prompt (streaming)."""
        modified_request = AgentRequest(
            prompt=f"{self._prefix}\n\n{request.prompt}",
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
            attachments=request.attachments,
            metadata=request.metadata,
        )

        async for chunk in self._client.send_request_stream(modified_request):
            yield chunk
```

---

### Example 3: Response Transformation Middleware

```python
class UppercaseMiddleware(BaseAgentClient):
    """Middleware that uppercases all responses."""

    def __init__(self, client: BaseAgentClient):
        super().__init__()
        self._client = client

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Uppercase response content."""
        response = await self._client.send_request(request)

        return AgentResponse(
            content=response.content.upper(),
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            metadata=response.metadata,
        )

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Uppercase stream chunks."""
        async for chunk in self._client.send_request_stream(request):
            yield StreamChunk(
                delta=chunk.delta.upper(),
                is_final=chunk.is_final,
                metadata=chunk.metadata,
            )
```

---

### Example 4: Stateful Middleware

```python
class RequestCounterMiddleware(BaseAgentClient):
    """Middleware that counts requests."""

    def __init__(self, client: BaseAgentClient):
        super().__init__()
        self._client = client
        self._count = 0
        self._lock = asyncio.Lock()  # Thread-safe counter

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Increment counter and delegate."""
        async with self._lock:
            self._count += 1
            request_id = self._count

        response = await self._client.send_request(request)

        # Add request ID to metadata
        response.metadata["request_id"] = request_id
        return response

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Increment counter for streaming."""
        async with self._lock:
            self._count += 1
            request_id = self._count

        async for chunk in self._client.send_request_stream(request):
            # Add ID to final chunk
            if chunk.is_final:
                chunk.metadata["request_id"] = request_id
            yield chunk

    @property
    def count(self) -> int:
        """Get current count."""
        return self._count
```

---

### Example 5: Using the Scaffolding CLI

```bash
# Generate middleware boilerplate
python -m madakit.cli.scaffold middleware MyCustomMiddleware --output my_middleware.py

# Generate test template
python -m madakit.cli.scaffold test MyCustomMiddleware --output test_my_middleware.py
```

---

## Testing Your Extensions

### Testing Providers

```python
import pytest
from madakit import AgentRequest

@pytest.mark.asyncio
async def test_custom_provider_request():
    """Test basic request/response."""
    client = CustomCloudClient(api_key="test-key")

    request = AgentRequest(prompt="Test prompt")
    # Mock HTTP calls here using httpx.MockTransport
    response = await client.send_request(request)

    assert response.content
    assert response.model
    assert response.total_tokens > 0

@pytest.mark.asyncio
async def test_custom_provider_streaming():
    """Test streaming."""
    client = CustomCloudClient(api_key="test-key")

    request = AgentRequest(prompt="Stream test")
    chunks = []

    async for chunk in client.send_request_stream(request):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert chunks[-1].is_final

@pytest.mark.asyncio
async def test_custom_provider_repr():
    """Test API key redaction in repr."""
    client = CustomCloudClient(api_key="sk-secret")
    repr_str = repr(client)

    assert "sk-secret" not in repr_str
    assert "***" in repr_str
```

---

### Testing Middleware

```python
from madakit.testing.utils import MockProvider

@pytest.mark.asyncio
async def test_logging_middleware():
    """Test logging middleware."""
    import logging
    from io import StringIO

    # Capture logs
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    logger = logging.getLogger("test")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Mock provider
    mock = MockProvider(
        responses=[AgentResponse(content="mock", model="test", input_tokens=10, output_tokens=5)]
    )

    # Wrap with logging
    client = SimpleLoggingMiddleware(mock, logger=logger)

    request = AgentRequest(prompt="Test")
    await client.send_request(request)

    # Check logs
    logs = log_stream.getvalue()
    assert "Request:" in logs
    assert "Response:" in logs

@pytest.mark.asyncio
async def test_middleware_preserves_fields():
    """Test middleware doesn't lose data."""
    mock = MockProvider(
        responses=[AgentResponse(
            content="test",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            metadata={"custom": "value"}
        )]
    )

    client = SimpleLoggingMiddleware(mock)

    request = AgentRequest(
        prompt="Test",
        system_prompt="System",
        max_tokens=200,
        temperature=0.8,
        metadata={"request_id": "123"}
    )

    response = await client.send_request(request)

    # Verify all fields preserved
    assert response.content == "test"
    assert response.model == "gpt-4o"
    assert response.input_tokens == 100
    assert response.output_tokens == 50
    assert response.metadata["custom"] == "value"

@pytest.mark.asyncio
async def test_middleware_stacking():
    """Test multiple middleware layers."""
    mock = MockProvider(responses=[AgentResponse(content="test", model="m", input_tokens=1, output_tokens=1)])

    # Stack middleware
    client = (
        RequestCounterMiddleware(
            UppercaseMiddleware(
                PromptPrefixMiddleware(
                    mock,
                    prefix="PREFIX:"
                )
            )
        )
    )

    request = AgentRequest(prompt="hello")
    response = await client.send_request(request)

    # Check transformations applied
    assert response.content == "TEST"  # Uppercased
    assert "request_id" in response.metadata  # Counter added ID
```

---

## Publishing Extensions

### Step 1: Package Structure

```
my-madakit-extension/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── madakit_extension/
│       ├── __init__.py
│       ├── providers/
│       │   └── custom.py
│       └── middleware/
│           └── custom.py
└── tests/
    ├── test_providers.py
    └── test_middleware.py
```

### Step 2: pyproject.toml

```toml
[project]
name = "madakit-extension-custom"
version = "0.1.0"
description = "Custom providers and middleware for madakit"
dependencies = [
    "madakit>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Step 3: __init__.py

```python
"""Custom extensions for madakit."""

from madakit_extension.providers.custom import CustomCloudClient
from madakit_extension.middleware.custom import CustomMiddleware

__all__ = [
    "CustomCloudClient",
    "CustomMiddleware",
]

__version__ = "0.1.0"
```

### Step 4: README.md

````markdown
# madakit-extension-custom

Custom providers and middleware for [madakit](https://github.com/yourusername/madakit).

## Installation

```bash
pip install madakit-extension-custom
```

## Usage

```python
from madakit_extension import CustomCloudClient

client = CustomCloudClient(api_key="...")
response = await client.send_request(request)
```

## License

MIT
````

### Step 5: Publish to PyPI

```bash
# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

---

## Best Practices

### 1. **Follow the ABC Contract**
Always implement required methods, delegate properly.

### 2. **Preserve Request/Response Fields**
Don't drop metadata, attachments, or other fields.

### 3. **Handle Errors Gracefully**
Use `ProviderError` / `MiddlewareError`, preserve stack traces.

### 4. **Test Thoroughly**
Unit tests, integration tests, edge cases.

### 5. **Document Your Extension**
Docstrings, README, usage examples.

### 6. **Redact Secrets**
Never log or repr API keys, passwords.

### 7. **Make Dependencies Optional**
Use deferred imports for external libraries.

### 8. **Use Type Annotations**
Full type hints for mypy compliance.

### 9. **Version Carefully**
Follow semver, test against madakit versions.

### 10. **Contribute Upstream**
Consider contributing useful extensions to core library.

---

## Next Steps

- [Architecture Guide](architecture.md) — Understand the design
- [User Guide](user-guide.md) — Learn usage patterns
- [API Reference](api-reference.md) — Complete API docs
- [Tutorial](tutorial.md) — Hands-on examples
