# madakit — Expansion Plan

```yaml
Document ID: MADAKIT-EXPAND-001
Version: 0.1.0
Status: APPROVED
Created: 2026-03-11
Classification: EXPANSION - Feature expansion for madakit post-Phase 7
Project: madakit (formerly mada-modelkit)
Author: Akil Abderrahim [Founder]
Parent Documents:
  - ARCHITECTURE.md (MODELKIT-ARCH-001)
  - IMPLEMENTATION.md (MODELKIT-IMPL-001)
  - AUDIT.md (Pre-release audit)
```

**Vision Change:** Release postponed. Expansion prioritized to build comprehensive feature set before v1.0.0 release.

---

## Table of Contents

1. [Overview](#overview)
2. [Expansion Phases](#expansion-phases)
3. [Phase 8: Production Middleware](#phase-8-production-middleware)
4. [Phase 9: Observability & Logging](#phase-9-observability--logging)
5. [Phase 10: Configuration System](#phase-10-configuration-system)
6. [Phase 11: Advanced Middleware](#phase-11-advanced-middleware)
7. [Phase 12: Tool Calling & Functions](#phase-12-tool-calling--functions)
8. [Phase 13: Batching & Aggregation](#phase-13-batching--aggregation)
9. [Phase 14: Additional Providers](#phase-14-additional-providers)
10. [Phase 15: Framework Integrations](#phase-15-framework-integrations)
11. [Phase 16: Developer Tools](#phase-16-developer-tools)
12. [Phase 17: Final Packaging & Release](#phase-17-final-packaging--release)

---

## Overview

### Current State (Post-Phase 7)
- **Foundation:** Types, errors, ABC ✅
- **Middleware:** 5 core middleware (retry, circuit breaker, cache, tracking, fallback) ✅
- **Providers:** 10 providers (4 cloud, 3 local, 3 native) ✅
- **Tests:** 1,168 tests, 100% pass rate ✅
- **Status:** Code complete, mypy issues pending

### Expansion Objectives
1. **Production hardening** - Rate limiting, cost control, timeouts, advanced resilience
2. **Observability** - Metrics, structured logging, tracing, monitoring
3. **Configuration** - Declarative middleware stacks, environment management
4. **Advanced features** - Tool calling, batching, multi-provider patterns
5. **Ecosystem integration** - LangChain, LlamaIndex, web frameworks
6. **Developer experience** - Scaffolding, testing utilities, migration tools

### Target Release
- **Version:** v1.0.0 (not v0.1.0)
- **Timeline:** TBD based on expansion completion
- **Documentation:** Comprehensive docs written post-expansion

---

## Expansion Phases

| Phase | Category | Description | Specs | Estimated Tasks | Dependencies |
|-------|----------|-------------|-------|-----------------|--------------|
| 8 | Production Middleware | Rate limiting, cost control, timeout | 3 | 18 | Phase 1-2 |
| 9 | Observability & Logging | Structured logging, metrics, OpenTelemetry | 3 | 15 | Phase 1-2 |
| 10 | Configuration System | YAML/JSON configs, validation, hot-reload | 2 | 12 | Phase 1-2 |
| 11 | Advanced Middleware | A/B testing, content filtering, templates | 4 | 20 | Phase 1-2, 8 |
| 12 | Tool Calling & Functions | Tool registry, function calling, workflows | 3 | 18 | Phase 1-2 |
| 13 | Batching & Aggregation | Request batching, consensus, stream aggregation | 3 | 15 | Phase 1-2 |
| 14 | Additional Providers | More cloud/local/specialized providers | 6 | 24 | Phase 3 |
| 15 | Framework Integrations | LangChain, LlamaIndex, FastAPI, Flask | 4 | 16 | All previous |
| 16 | Developer Tools | Scaffolding, testing utils, migration tools | 3 | 12 | All previous |
| 17 | Final Packaging & Release | Docs, LICENSE, README, CI/CD | 5 | 15 | All previous |
| **Total** | | | **36** | **165** | |

---

## Phase 8: Production Middleware

**Objective:** Production-grade resilience and resource management middleware.

### Spec 8.1: RateLimitMiddleware (`middleware/rate_limit.py`)

**Purpose:** Token bucket / leaky bucket rate limiting for API quota management.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 8.1.1 | Constructor | `client`, `requests_per_second=10.0`, `burst_size=None`, `strategy="token_bucket"` |
| 8.1.2 | Token bucket algorithm | Accumulate tokens at rate; consume on request; block when empty |
| 8.1.3 | Leaky bucket algorithm | Queue requests; process at fixed rate; drop when queue full |
| 8.1.4 | `send_request` with rate limit | Await token availability before delegation |
| 8.1.5 | `send_request_stream` with rate limit | Same token consumption as send_request |
| 8.1.6 | Per-key rate limiting | Optional `key_fn` for per-user/per-endpoint limits |
| 8.1.7 | Tests for `rate_limit.py` | Token bucket accuracy, leaky bucket queue, burst handling, key-based isolation |

### Spec 8.2: CostControlMiddleware (`middleware/cost_control.py`)

**Purpose:** Budget tracking, spending alerts, cost caps.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 8.2.1 | Constructor | `client`, `budget_cap=None`, `alert_threshold=0.8`, `cost_fn`, `on_alert=None` |
| 8.2.2 | Budget tracking | Accumulate costs via `cost_fn`; track spending against cap |
| 8.2.3 | Budget cap enforcement | Raise `BudgetExceededError` when cap reached |
| 8.2.4 | Alert callbacks | Call `on_alert(current, threshold)` when threshold crossed |
| 8.2.5 | `send_request` with cost tracking | Increment spend after successful request |
| 8.2.6 | Cost reset | `reset_budget()` method to restart tracking period |
| 8.2.7 | Tests for `cost_control.py` | Cap enforcement, alert firing, reset, cost accumulation |

### Spec 8.3: TimeoutMiddleware (`middleware/timeout.py`)

**Purpose:** Request-level timeout enforcement (separate from HTTP timeouts).

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 8.3.1 | Constructor | `client`, `timeout_seconds=30.0` |
| 8.3.2 | `send_request` with timeout | `asyncio.wait_for` wrapper; raise `TimeoutError` on expiry |
| 8.3.3 | `send_request_stream` with timeout | Timeout applies to first chunk arrival only |
| 8.3.4 | Tests for `timeout.py` | Timeout triggers, successful fast requests, stream behavior |

---

## Phase 9: Observability & Logging

**Objective:** Structured logging, metrics export, distributed tracing.

### Spec 9.1: LoggingMiddleware (`middleware/logging.py`)

**Purpose:** Structured logging (JSON logs) for requests, responses, errors.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 9.1.1 | Constructor | `client`, `logger=None`, `log_level="INFO"`, `include_prompts=False` |
| 9.1.2 | Request logging | Log request start with ID, prompt (if enabled), metadata |
| 9.1.3 | Response logging | Log completion with duration, tokens, model |
| 9.1.4 | Error logging | Log exceptions with stack traces, context |
| 9.1.5 | Correlation IDs | Generate/propagate request IDs for tracing |
| 9.1.6 | Tests for `logging.py` | Log output verification, ID propagation, PII filtering |

### Spec 9.2: MetricsMiddleware (`middleware/metrics.py`)

**Purpose:** Prometheus-compatible metrics export.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 9.2.1 | Constructor | `client`, `registry=None`, `prefix="madakit"` |
| 9.2.2 | Counter metrics | Total requests, errors by type, provider calls |
| 9.2.3 | Histogram metrics | Latency distribution, token counts, TTFT |
| 9.2.4 | Gauge metrics | Active requests, circuit breaker state |
| 9.2.5 | Label support | Labels for provider, model, status |
| 9.2.6 | Tests for `metrics.py` | Metric increments, histogram buckets, label cardinality |

### Spec 9.3: OpenTelemetryMiddleware (`middleware/opentelemetry.py`)

**Purpose:** Distributed tracing with OpenTelemetry.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 9.3.1 | Constructor | `client`, `tracer=None`, `service_name="madakit"` |
| 9.3.2 | Span creation | Create span per request with attributes (model, provider, tokens) |
| 9.3.3 | Context propagation | Propagate trace context across middleware |
| 9.3.4 | Error recording | Record exceptions as span events |
| 9.3.5 | Tests for `opentelemetry.py` | Span creation, attribute setting, context propagation |

---

## Phase 10: Configuration System

**Objective:** Declarative middleware stacks via YAML/JSON, environment management.

### Spec 10.1: Configuration Schema (`config/_schema.py`)

**Purpose:** Define configuration data structures.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 10.1.1 | ProviderConfig dataclass | `type`, `model`, `api_key`, `base_url`, `**kwargs` |
| 10.1.2 | MiddlewareConfig dataclass | `type`, `params: dict[str, Any]` |
| 10.1.3 | StackConfig dataclass | `provider`, `middleware: list[MiddlewareConfig]` |
| 10.1.4 | Validation | Validate required fields, types, middleware order |
| 10.1.5 | Tests for `_schema.py` | Construction, validation, error messages |

### Spec 10.2: Configuration Loader (`config/loader.py`)

**Purpose:** Load configs from YAML/JSON, instantiate stacks.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 10.2.1 | YAML parser | Parse YAML to StackConfig |
| 10.2.2 | JSON parser | Parse JSON to StackConfig |
| 10.2.3 | Environment variable substitution | `${ENV_VAR}` syntax in configs |
| 10.2.4 | Stack builder | Instantiate provider + middleware from config |
| 10.2.5 | Config validation | Validate before instantiation |
| 10.2.6 | Error handling | Clear error messages for invalid configs |
| 10.2.7 | Tests for `loader.py` | YAML/JSON parsing, env substitution, stack creation, error cases |

---

## Phase 11: Advanced Middleware

**Objective:** Sophisticated middleware for A/B testing, content filtering, templating.

### Spec 11.1: ABTestMiddleware (`middleware/ab_test.py`)

**Purpose:** Split traffic between providers for A/B testing.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 11.1.1 | Constructor | `variants: list[tuple[BaseAgentClient, float]]`, `key_fn=None` |
| 11.1.2 | Traffic splitting | Deterministic split based on hash of key |
| 11.1.3 | Variant selection | Select provider based on weight distribution |
| 11.1.4 | Metadata tagging | Add `variant` to response metadata |
| 11.1.5 | Tests for `ab_test.py` | Distribution accuracy, determinism, metadata |

### Spec 11.2: ContentFilterMiddleware (`middleware/content_filter.py`)

**Purpose:** PII redaction, safety checks, content moderation.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 11.2.1 | Constructor | `client`, `redact_pii=True`, `safety_check=None` |
| 11.2.2 | PII detection | Regex-based email, SSN, credit card detection |
| 11.2.3 | Redaction | Replace PII with `[REDACTED]` |
| 11.2.4 | Safety callbacks | Call `safety_check(prompt)` before request |
| 11.2.5 | Response filtering | Filter responses for harmful content |
| 11.2.6 | Tests for `content_filter.py` | PII detection accuracy, redaction, safety blocking |

### Spec 11.3: PromptTemplateMiddleware (`middleware/prompt_template.py`)

**Purpose:** Template management, variable injection.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 11.3.1 | Constructor | `client`, `templates: dict[str, str]` |
| 11.3.2 | Template rendering | Jinja2-style variable substitution |
| 11.3.3 | Template registry | Store and retrieve templates by name |
| 11.3.4 | Variable validation | Ensure all variables provided |
| 11.3.5 | Tests for `prompt_template.py` | Rendering, variable substitution, errors |

### Spec 11.4: LoadBalancingMiddleware (`middleware/load_balancing.py`)

**Purpose:** Weighted routing, health-based distribution.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 11.4.1 | Constructor | `providers: list[tuple[BaseAgentClient, float]]`, `strategy="weighted"` |
| 11.4.2 | Weighted round-robin | Select provider based on weights |
| 11.4.3 | Health-based routing | Skip unhealthy providers |
| 11.4.4 | Least-latency routing | Route to fastest provider |
| 11.4.5 | Tests for `load_balancing.py` | Weight distribution, health skipping, latency routing |

---

## Phase 12: Tool Calling & Functions

**Objective:** Function calling, tool registry, multi-step workflows.

### Spec 12.1: Tool Registry (`tools/registry.py`)

**Purpose:** Register and manage callable tools.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 12.1.1 | Tool dataclass | `name`, `description`, `parameters: dict`, `function: Callable` |
| 12.1.2 | Registry class | `register()`, `get()`, `list_tools()` |
| 12.1.3 | OpenAPI schema generation | Convert tool to OpenAPI function schema |
| 12.1.4 | Parameter validation | Validate inputs against schema |
| 12.1.5 | Tests for `registry.py` | Registration, retrieval, schema generation |

### Spec 12.2: FunctionCallingMiddleware (`middleware/function_calling.py`)

**Purpose:** Automatic function calling on tool use.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 12.2.1 | Constructor | `client`, `registry: ToolRegistry`, `max_iterations=3` |
| 12.2.2 | Tool detection | Parse response for tool calls |
| 12.2.3 | Tool execution | Call registered functions |
| 12.2.4 | Result injection | Inject results back into conversation |
| 12.2.5 | Iteration limiting | Prevent infinite loops |
| 12.2.6 | Tests for `function_calling.py` | Tool detection, execution, result injection |

### Spec 12.3: WorkflowEngine (`tools/workflow.py`)

**Purpose:** Multi-step agent workflows.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 12.3.1 | Step dataclass | `name`, `client`, `condition: Callable`, `tools: list` |
| 12.3.2 | Workflow class | `add_step()`, `execute()` |
| 12.3.3 | Conditional branching | Execute steps based on conditions |
| 12.3.4 | State management | Pass state between steps |
| 12.3.5 | Error handling | Graceful failure, rollback |
| 12.3.6 | Tests for `workflow.py` | Multi-step execution, branching, state passing |

---

## Phase 13: Batching & Aggregation

**Objective:** Request batching, multi-provider consensus, stream aggregation.

### Spec 13.1: BatchingMiddleware (`middleware/batching.py`)

**Purpose:** Collect multiple requests, send as batch.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 13.1.1 | Constructor | `client`, `batch_size=10`, `max_wait_ms=100` |
| 13.1.2 | Request buffering | Collect requests until batch_size or timeout |
| 13.1.3 | Batch dispatch | Send batch to provider |
| 13.1.4 | Response distribution | Return responses to original callers |
| 13.1.5 | Tests for `batching.py` | Batch collection, timeout, response matching |

### Spec 13.2: ConsensusMiddleware (`middleware/consensus.py`)

**Purpose:** Send to multiple providers, vote/aggregate results.

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 13.2.1 | Constructor | `providers: list[BaseAgentClient]`, `strategy="majority"` |
| 13.2.2 | Parallel dispatch | Send to all providers concurrently |
| 13.2.3 | Majority voting | Return most common response |
| 13.2.4 | Confidence scoring | Aggregate confidence scores |
| 13.2.5 | Tests for `consensus.py` | Voting accuracy, parallel execution |

### Spec 13.3: StreamAggregationMiddleware (`middleware/stream_aggregation.py`)

**Purpose:** Combine multiple streams (e.g., mix providers).

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 13.3.1 | Constructor | `clients: list[BaseAgentClient]`, `strategy="merge"` |
| 13.3.2 | Stream merging | Interleave chunks from multiple streams |
| 13.3.3 | Race mode | Yield from first stream, cancel others |
| 13.3.4 | Tests for `stream_aggregation.py` | Merge correctness, race behavior |

---

## Phase 14: Additional Providers

**Objective:** Expand provider coverage (cloud, local, specialized).

### Spec 14.1: Additional Cloud Providers

| Provider | Base URL | Notes |
|----------|----------|-------|
| CohereClient | https://api.cohere.ai/v1 | Custom payload format |
| MistralClient | https://api.mistral.ai/v1 | OpenAI-compatible |
| TogetherClient | https://api.together.xyz/v1 | OpenAI-compatible |
| GroqClient | https://api.groq.com/openai/v1 | OpenAI-compatible |
| ReplicateClient | https://api.replicate.com/v1 | Custom format |
| FireworksClient | https://api.fireworks.ai/inference/v1 | OpenAI-compatible |

**Task allocation:** 4 tasks per provider (constructor, tests, format handling, streaming)

### Spec 14.2: Additional Local Providers

| Provider | Default URL | Notes |
|----------|-------------|-------|
| LMStudioClient | http://localhost:1234/v1 | OpenAI-compatible |
| JanClient | http://localhost:1337/v1 | OpenAI-compatible |
| GPT4AllClient | http://localhost:4891/v1 | OpenAI-compatible |

**Task allocation:** 3 tasks per provider

### Spec 14.3: Specialized Providers

| Provider | Purpose | Notes |
|----------|---------|-------|
| StabilityAIClient | Image generation | Multimodal support |
| ElevenLabsClient | Audio/TTS | Custom format |
| EmbeddingProvider | Embedding-specific | Optimized for embeddings |

**Task allocation:** 4 tasks per provider

---

## Phase 15: Framework Integrations

**Objective:** Integrate with LangChain, LlamaIndex, web frameworks.

### Spec 15.1: LangChain Integration (`integrations/langchain.py`)

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 15.1.1 | LangChain LLM wrapper | Wrap madakit client as LangChain LLM |
| 15.1.2 | Callback integration | Bridge LangChain callbacks to madakit middleware |
| 15.1.3 | Streaming support | LangChain streaming via madakit streams |
| 15.1.4 | Tests | LangChain chain execution with madakit |

### Spec 15.2: LlamaIndex Integration (`integrations/llamaindex.py`)

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 15.2.1 | LlamaIndex LLM wrapper | Wrap madakit client as LlamaIndex LLM |
| 15.2.2 | Embedding support | Madakit embedding providers for LlamaIndex |
| 15.2.3 | Tests | LlamaIndex query engine with madakit |

### Spec 15.3: FastAPI Integration (`integrations/fastapi.py`)

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 15.3.1 | Dependency injection | FastAPI dependency for madakit clients |
| 15.3.2 | Streaming responses | SSE streaming from madakit to FastAPI |
| 15.3.3 | Middleware integration | FastAPI middleware → madakit logging |
| 15.3.4 | Tests | FastAPI endpoint with madakit client |

### Spec 15.4: Flask Integration (`integrations/flask.py`)

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 15.4.1 | Flask extension | Flask extension for madakit |
| 15.4.2 | Request context | Bind madakit client to request context |
| 15.4.3 | Streaming | Server-sent events from madakit |
| 15.4.4 | Tests | Flask route with madakit client |

---

## Phase 16: Developer Tools

**Objective:** Scaffolding, testing utilities, migration tools.

### Spec 16.1: Scaffolding CLI (`cli/scaffold.py`)

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 16.1.1 | Provider template generator | Generate custom provider boilerplate |
| 16.1.2 | Middleware template generator | Generate custom middleware boilerplate |
| 16.1.3 | Test template generator | Generate test file templates |
| 16.1.4 | CLI interface | `madakit scaffold provider <name>` |

### Spec 16.2: Testing Utilities (`testing/utils.py`)

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 16.2.1 | MockProvider enhancements | Configurable latency, error injection |
| 16.2.2 | Assertion helpers | `assert_cache_hit()`, `assert_retry_count()` |
| 16.2.3 | Fixture library | Pytest fixtures for common scenarios |
| 16.2.4 | Tests | Utilities tested |

### Spec 16.3: Migration Tools (`cli/migrate.py`)

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 16.3.1 | LangChain migrator | Convert LangChain code to madakit |
| 16.3.2 | Config converter | Convert other formats to madakit configs |
| 16.3.3 | Compatibility checker | Check if migration is feasible |
| 16.3.4 | Tests | Migration accuracy |

---

## Phase 17: Final Packaging & Release

**Objective:** Documentation, release artifacts, CI/CD, v1.0.0 release.

### Spec 17.1: Comprehensive Documentation

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 17.1.1 | Architecture guide | Hand-written guide explaining layers |
| 17.1.2 | User guide | Provider selection, middleware config |
| 17.1.3 | API reference | Comprehensive API docs |
| 17.1.4 | Tutorial/Cookbook | Real examples, patterns |
| 17.1.5 | Extension guide | Building custom providers/middleware |

### Spec 17.2: Release Artifacts

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 17.2.1 | LICENSE | MIT license file |
| 17.2.2 | README.md | Installation, quickstart, features |
| 17.2.3 | CHANGELOG.md | Full version history |
| 17.2.4 | CONTRIBUTING.md | Contribution guidelines |

### Spec 17.3: CI/CD

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 17.3.1 | GitHub Actions | Test workflow on push/PR |
| 17.3.2 | Coverage reporting | Codecov integration |
| 17.3.3 | Release automation | Automated PyPI publishing |

### Spec 17.4: Documentation Hosting

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 17.4.1 | MkDocs setup | Configure MkDocs |
| 17.4.2 | Read the Docs | Deploy to readthedocs.io |

### Spec 17.5: Final Release

| Task | Description | Acceptance Criteria |
|------|-------------|-------------------|
| 17.5.1 | Version bump | Set version to 1.0.0 |
| 17.5.2 | Git tag | Tag v1.0.0 |
| 17.5.3 | PyPI publish | Upload to PyPI |
| 17.5.4 | Announcement | Blog post, social media |

---

## Summary

**Total Expansion Effort:**
- **10 new phases** (8-17)
- **36 specs**
- **~165 tasks**
- **Estimated duration:** 4-8 weeks (depending on parallel work)

**Key Milestones:**
1. **Phase 8-9:** Production-ready (rate limiting, cost control, observability)
2. **Phase 10-11:** Advanced features (config system, advanced middleware)
3. **Phase 12-13:** Cutting-edge (tool calling, batching, consensus)
4. **Phase 14:** Comprehensive provider coverage
5. **Phase 15-16:** Ecosystem integration & developer tools
6. **Phase 17:** v1.0.0 release

**Release Target:** madakit v1.0.0 with comprehensive feature set and documentation.
