# Development Notes

## Current Scope

This workspace contains a fresh implementation of a lightweight interactive pandas AI helper in `pandas_ai/`.

Public API:

- `setup_ai(...)`
- `ask_ai(text, dfs=None)`

Reference repo:

- `repo_inspect/` is a clone of the original `DashyDashOrg/pandas-llm` project and is kept only for comparison.

## Current Architecture

Core package layout:

- `pandas_ai/api.py`
  - Public entrypoints.
  - Stores the singleton session.
- `pandas_ai/session.py`
  - Orchestrates prompt building, backend invocation, and response parsing.
- `pandas_ai/config.py`
  - Resolves defaults and environment variables.
- `pandas_ai/schema.py`
  - Normalizes dataframe inputs and builds dataframe schema context.
  - Supports `dfs=None`, a single dataframe, a sequence of dataframes, or a mapping of names to dataframes.
- `pandas_ai/prompting.py`
  - Builds the user prompt from dataframe schema and user request.
- `pandas_ai/parsing.py`
  - Extracts code from plain text or fenced code blocks.
- `pandas_ai/backends/anthropic.py`
  - Anthropic Messages API backend.
- `pandas_ai/backends/openai_compat.py`
  - OpenAI-compatible backend for LM Studio and similar local servers.

## Status

Implemented:

- Default backend is `claude` via Anthropic Messages API.
- Secondary backend is `lmstudio` via OpenAI-compatible HTTP API.
- Environment variable support:
  - `PANDAS_AI_BACKEND`
  - `PANDAS_AI_MODEL`
  - `PANDAS_AI_TIMEOUT`
  - `PANDAS_AI_BASE_URL`
  - `PANDAS_AI_SYSTEM_PROMPT`
  - `ANTHROPIC_API_KEY`
- `ask_ai()` returns a short backend note plus copy/paste-ready code text.
- `ask_ai(dfs=None)` attempts to discover a caller-side `df` from the Python stack.
- Unit tests cover config, backend payload shape, prompt assembly, parsing, and dataframe normalization.

Current limitations:

- No packaging metadata yet beyond the minimal module layout.
- No example scripts yet for actual REPL usage.
- No conversation memory.
- No automatic code execution or sandbox.
- No local transformer backend implementation yet.
- The current environment used during implementation did not have `pandas` installed, so tests use dataframe-like dummy objects.

## Testing

Primary test command used:

```bash
python -m unittest tests.test_pandas_ai -v
```

Additional syntax verification used:

```bash
python -m py_compile pandas_ai/__init__.py pandas_ai/api.py pandas_ai/config.py pandas_ai/errors.py pandas_ai/schema.py pandas_ai/prompting.py pandas_ai/parsing.py pandas_ai/session.py pandas_ai/backends/__init__.py pandas_ai/backends/base.py pandas_ai/backends/anthropic.py pandas_ai/backends/openai_compat.py tests/test_pandas_ai.py
```

## Notes For Next Pass

When switching to a Linux workspace with `pandas` installed, the next useful steps are:

1. Replace or supplement dummy dataframe tests with real pandas DataFrame tests.
2. Add a small end-to-end example script for REPL / IPython usage.
3. Add packaging files if this should become installable.
4. Decide whether `ask_ai(dfs=None)` stack inspection is acceptable long-term or should be replaced by an explicit namespace argument.
5. Add an optional local backend implementation if air-gapped use is a first-class target.

## Git / Workspace Notes

- The current workspace root is `C:\PL\Dropbox\3_my_program\experimenting\pandas_llm`.
- `repo_inspect/` is third-party reference code and should not be modified unless intentionally syncing or comparing behavior.
