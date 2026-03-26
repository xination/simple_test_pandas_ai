# pandas_ai

Lightweight interactive pandas AI helper with two public functions:

- `setup_ai()`
- `ask_ai()`

Default backend is Anthropic Messages API (`backend="claude"`). LM Studio is supported through an OpenAI-compatible backend.

## Minimal usage

```python
from pandas_ai import setup_ai, ask_ai

setup_ai(backend="claude", api_key="YOUR_ANTHROPIC_KEY")
print(ask_ai("show the first 5 rows", df))
```

```python
from pandas_ai import setup_ai, ask_ai

setup_ai(backend="lmstudio", base_url="http://127.0.0.1:1234/v1", model="local-model")
print(ask_ai("join df0 and df1 on user_id", [df0, df1]))
```

## Development

See `DEVELOPMENT.md` for architecture, status, and next-step notes.
