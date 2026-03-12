# Configuration

madakit supports declarative configuration via YAML or JSON files with environment variable substitution.

## Quick Example

```yaml
# config.yaml
provider:
  type: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}

middleware:
  - type: cache
    params:
      ttl_seconds: 3600
  - type: retry
    params:
      max_attempts: 3
  - type: timeout
    params:
      timeout_seconds: 30.0
```

```python
from madakit.config import ConfigLoader

loader = ConfigLoader()
client = loader.from_yaml("config.yaml")
response = await client.send_request(request)
```

## Environment Variables

Use `${VAR}` or `${VAR:default}` syntax:

```yaml
provider:
  type: openai
  api_key: ${OPENAI_API_KEY}
  base_url: ${OPENAI_BASE_URL:https://api.openai.com/v1}
```

## Configuration Schema

### Provider Configuration

```yaml
provider:
  type: <provider_type>  # Required: openai, anthropic, ollama, etc.
  model: <model_name>    # Optional: provider default used if omitted
  api_key: <key>         # Required for cloud providers
  base_url: <url>        # Optional: override default endpoint
  kwargs:                # Optional: provider-specific parameters
    temperature: 0.7
    max_tokens: 100
```

### Middleware Configuration

```yaml
middleware:
  - type: <middleware_type>  # retry, cache, circuit-breaker, etc.
    params:                  # Middleware-specific parameters
      <param>: <value>
```

## Supported Providers

| Type | Description |
|------|-------------|
| `openai` | OpenAI GPT models |
| `anthropic` | Anthropic Claude models |
| `gemini` | Google Gemini models |
| `deepseek` | DeepSeek models |
| `ollama` | Ollama local server |
| `vllm` | vLLM inference server |
| `transformers` | Hugging Face Transformers |
| `llamacpp` | llama.cpp Python bindings |
| `cohere`, `mistral`, `together`, `groq`, `fireworks`, `replicate` | Other cloud providers |

## Supported Middleware

| Type | Key Parameters |
|------|----------------|
| `retry` | `max_attempts`, `base_delay_ms`, `max_delay_ms` |
| `cache` | `ttl_seconds`, `max_size` |
| `circuit-breaker` | `failure_threshold`, `recovery_timeout_s` |
| `tracking` | `cost_per_input_token`, `cost_per_output_token` |
| `fallback` | (not configurable, use programmatic API) |
| `rate-limit` | `requests_per_second`, `burst_size` |
| `cost-control` | `max_cost`, `alert_threshold` |
| `timeout` | `timeout_seconds` |
| `logging` | `log_level`, `include_prompts` |
| `metrics` | `prefix`, `track_labels` |

## Complete Example

```yaml
# production.yaml
provider:
  type: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
  kwargs:
    temperature: 0.7
    max_tokens: 500

middleware:
  - type: cache
    params:
      ttl_seconds: 3600
      max_size: 1000
  - type: circuit-breaker
    params:
      failure_threshold: 5
      recovery_timeout_s: 60
  - type: retry
    params:
      max_attempts: 3
      base_delay_ms: 1000
      max_delay_ms: 10000
  - type: tracking
    params:
      cost_per_input_token: 0.00001
      cost_per_output_token: 0.00003
  - type: timeout
    params:
      timeout_seconds: 30.0
  - type: logging
    params:
      log_level: INFO
      include_prompts: false
```

## JSON Format

JSON format is also supported:

```json
{
  "provider": {
    "type": "openai",
    "model": "gpt-4",
    "api_key": "${OPENAI_API_KEY}"
  },
  "middleware": [
    {"type": "cache", "params": {"ttl_seconds": 3600}},
    {"type": "retry", "params": {"max_attempts": 3}}
  ]
}
```

```python
loader = ConfigLoader()
client = loader.from_json("config.json")
```

## Programmatic Configuration

Use `from_dict()` for dynamic configuration:

```python
config = {
    "provider": {
        "type": "openai",
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY")
    },
    "middleware": [
        {"type": "cache", "params": {"ttl_seconds": 3600}},
        {"type": "retry", "params": {"max_attempts": 3}}
    ]
}

loader = ConfigLoader()
client = loader.from_dict(config)
```

## Environment-Aware Configuration

Use different configs per environment:

```python
import os
from madakit.config import ConfigLoader

env = os.getenv("ENV", "development")
config_file = f"config.{env}.yaml"

loader = ConfigLoader()
client = loader.from_yaml(config_file)
```

For detailed configuration documentation, see the [User Guide](../user-guide.md#configuration).
