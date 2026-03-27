# pandas_ai

Lightweight interactive pandas AI helper with two public functions:

- `setup_ai()`
- `ask_ai()`

Default backend is Anthropic Messages API (`backend="claude"`), with default model `claude-haiku-4-5-20251001`. LM Studio is supported through an OpenAI-compatible backend. Streaming is configured at `setup_ai()` time, enabled by default, and uses only the Python standard library.

## Minimal usage

```python
from pandas_ai import setup_ai, ask_ai

setup_ai(backend="claude", api_key="YOUR_ANTHROPIC_KEY")
result = ask_ai("show the first 5 rows", df)
print(result)
```

```python
from pandas_ai import setup_ai, ask_ai

setup_ai(backend="lmstudio", base_url="http://127.0.0.1:1234/v1", model="local-model", stream=False)
print(ask_ai("join df0 and df1 on user_id", [df0, df1]))
```

When `stream=True`, `ask_ai()` still returns the final value. Use `stream_output=True` to print streamed chunks to stdout, or pass a custom `stream_handler`. The default handler is silent, so REPL usage does not print duplicate output. Use `stream_delay` if you want to slow chunk display down for visual confirmation.

```python
setup_ai(stream=True, stream_output=True, stream_delay=0.03)
result = ask_ai("show the first 5 rows", df)
```

If you are running in plain Python interactive mode and want streamed output without the final result being echoed again, enable the REPL helper once:

```python
from pandas_ai import enable_ai_result_displayhook

enable_ai_result_displayhook()
```

When `output_format="json"`, partial chunks are not expected to be valid JSON. Parsing happens only after the complete response is assembled.

## Development

See `DEVELOPMENT.md` for architecture, status, and next-step notes.

## Demo

A small Chicago housing style demo is available in `examples/chicago_housing_demo.py`.
A local Hugging Face streaming demo is available in `examples/example2_local_qwen_stream.py`.

```bash
export ANTHROPIC_API_KEY="your_api_key"
python examples/chicago_housing_demo.py
```

The demo builds a tiny in-memory dataset with columns such as `community_area`,
`housing_type`, `sqft`, and `list_price`, then asks the model to generate pandas
code for a simple housing analysis task.
