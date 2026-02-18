# mada-modelkit — Claude Code Instructions

## Project

mada-modelkit is a composable AI client library — cloud, local, and native providers behind one async interface, layered with middleware.

- **Architecture:** `ARCHITECTURE.md` (MODELKIT-ARCH-001, APPROVED)
- **Implementation plan:** `IMPLEMENTATION.md` (MODELKIT-IMPL-001, APPROVED)
- **Progress tracker:** `BUILD_JOURNAL.md`

## Workflow

The unit of work is a **task** (from IMPLEMENTATION.md). Each task is an atomic, testable piece of implementation.

### After completing each task:

1. **Run tests** for the affected file(s) — ensure they pass
2. **Git commit** the implementation + tests together with message format: `<type>(<scope>): <subject>`
   - Types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`
   - Scopes: `types`, `errors`, `base`, `retry`, `circuit-breaker`, `cache`, `tracking`, `fallback`, `middleware`, `http-base`, `openai-compat`, `openai`, `anthropic`, `gemini`, `deepseek`, `ollama`, `vllm`, `localai`, `llamacpp`, `transformers`, `packaging`
3. **Update BUILD_JOURNAL.md** — mark the task as `done` in the task table, add notes if relevant
4. **Stop and await instructions** — do not proceed to the next task without user confirmation

### Do NOT:

- Batch multiple tasks into one commit (one task = one commit)
- Proceed to the next task without being told to
- Skip tests — every task that produces code must have corresponding tests
- Modify ARCHITECTURE.md — it is the approved spec

## Code Quality

- Python 3.11+
- Dataclasses for types (not Pydantic — zero core deps requirement)
- Type annotations on all public functions and methods
- Ruff for linting (line-length=100)
- mypy strict mode
- No `print()` in library code
- Tests with pytest + pytest-asyncio
- **Docstrings required at every level.**
  - *Module* — every `.py` file opens with a docstring before imports. Library modules: state purpose and zero-dep constraint. Test modules: state what is under test and which aspects are covered.
  - *Class* — every class has a one-line docstring. Library classes: purpose. Test classes: the subject under test.
  - *Method/function* — every public function or method in `src/` has a docstring. Every test method in `tests/` has a one-sentence docstring stating what it asserts.

## Key Architectural Rules

- **Zero core dependencies.** `_types.py`, `_errors.py`, `_base.py`, and all middleware use only the standard library
- **Providers are optional extras.** Provider dependencies (httpx, llama-cpp-python, transformers) are optional. Use deferred imports
- **Middleware is blind to provider.** Middleware operates through the ABC contract only
- **Both paths mandatory.** Every middleware must implement `send_request` AND `send_request_stream`
- **Secure by default.** API keys redacted in `__repr__`, TLS enforced for cloud, no `eval`/`exec`

## Package Structure

```
src/mada_modelkit/       # library source
tests/                   # test files mirror src/ structure
ARCHITECTURE.md          # approved architecture (read-only reference)
IMPLEMENTATION.md        # phased build plan (read-only reference)
BUILD_JOURNAL.md         # progress tracker (update after each task)
```
