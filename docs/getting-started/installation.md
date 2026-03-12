# Installation

## Requirements

- Python 3.11 or higher
- pip

## Install from PyPI

### Core Library (Zero Dependencies)

```bash
pip install madakit
```

The core library includes:
- Type system (`AgentRequest`, `AgentResponse`, `StreamChunk`, etc.)
- Error hierarchy (`AgentError`, `ProviderError`, etc.)
- Abstract base client (`BaseAgentClient`)
- All 16 middleware implementations

**No external dependencies required!** The core uses only Python's standard library.

### With Cloud Providers

```bash
pip install madakit[cloud]
```

Installs `httpx>=0.27` for cloud API access. Includes providers:
- OpenAI, Anthropic, Gemini, DeepSeek, Cohere, Mistral, Together, Groq, Fireworks, Replicate

### With Local Server Providers

```bash
pip install madakit[local]
```

Installs `httpx>=0.27` for local server HTTP access. Includes providers:
- Ollama, vLLM, LocalAI, llama.cpp server, LM Studio, Jan, GPT4All

### With Native Providers

```bash
pip install madakit[native]
```

Installs:
- `transformers>=4.40` and `torch>=2.0` for Hugging Face models
- `llama-cpp-python>=0.2` for llama.cpp Python bindings

### With Metrics Support

```bash
pip install madakit[metrics]
```

Installs `prometheus-client>=0.20` for metrics middleware.

### Everything

```bash
pip install madakit[all]
```

Installs all optional dependencies (cloud, local, native, metrics, config).

## Development Installation

```bash
# Clone repository
git clone https://github.com/madahub/madakit.git
cd madakit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

The `dev` extra includes:
- `pytest>=8.0`, `pytest-asyncio>=0.23` — testing
- `ruff>=0.4` — linting and formatting
- `mypy>=1.10` — type checking
- All optional dependencies

## Verify Installation

```python
import madakit

print(madakit.__version__)  # Should print version number

# Test import
from madakit import AgentRequest, BaseAgentClient
```

## Optional Dependencies by Use Case

| Use Case | Extra | Dependencies |
|----------|-------|--------------|
| OpenAI, Anthropic, Gemini, etc. | `[cloud]` | `httpx>=0.27` |
| Ollama, vLLM, etc. | `[local]` | `httpx>=0.27` |
| Transformers | `[native]` | `transformers>=4.40`, `torch>=2.0` |
| llama.cpp | `[llamacpp]` | `llama-cpp-python>=0.2` |
| Prometheus metrics | `[metrics]` | `prometheus-client>=0.20` |
| YAML/JSON config | `[config]` | `pyyaml>=6.0` |

## Next Steps

- [Quickstart](quickstart.md) — Get started in 5 minutes
- [Tutorial](../tutorial.md) — Learn with examples
- [Provider Guide](providers.md) — Choose a provider
