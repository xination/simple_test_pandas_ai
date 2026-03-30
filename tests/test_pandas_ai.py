import io
import json
import os
import sys
import unittest
from unittest import mock
from urllib import error

from pandas_ai import ask_ai, setup_ai
from pandas_ai import api as api_module
from pandas_ai import interactive as interactive_module
from pandas_ai.backends.anthropic import AnthropicBackend
from pandas_ai.backends.openai_compat import OpenAICompatBackend
from pandas_ai.config import DEFAULT_BACKEND, DEFAULT_LMSTUDIO_BASE_URL, DEFAULT_LMSTUDIO_MODEL, DEFAULT_SYSTEM_PROMPT, load_config
from pandas_ai.errors import BackendError, ConfigurationError
from pandas_ai.parsing import extract_code
from pandas_ai.session import AIResult


class DummyResponse(object):
    def __init__(self, payload):
        self.payload = payload
        self._lines = None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def readline(self):
        if self._lines is None:
            self._lines = []
        if self._lines:
            return self._lines.pop(0)
        return b""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyStreamResponse(DummyResponse):
    def __init__(self, lines):
        super(DummyStreamResponse, self).__init__(payload=None)
        self._lines = [line.encode("utf-8") for line in lines]

    def read(self):
        raise AssertionError("stream responses should be consumed via readline()")


class FakeBackend(object):
    name = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return "```python\nresult = df['value'].sum()\n```"

    def stream_generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        for chunk in ("```python\n", "result = df['value'].sum()\n", "```"):
            yield chunk


class JsonBackend(object):
    name = "fake-json"

    def generate(self, system_prompt, user_prompt):
        return '{"ok": true, "rows": 3}'

    def stream_generate(self, system_prompt, user_prompt):
        for chunk in ('{"ok": ', 'true, ', '"rows": 3}'):
            yield chunk


class CollectingHandler(object):
    def __init__(self):
        self.chunks = []

    def __call__(self, chunk):
        self.chunks.append(chunk)


class DummyDataFrame(object):
    def __init__(self, data):
        self._data = data
        self.columns = list(data.keys())
        self.dtypes = {}
        row_count = 0
        for key, values in data.items():
            row_count = max(row_count, len(values))
            dtype_name = type(values[0]).__name__ if values else "object"
            self.dtypes[key] = dtype_name
        self.shape = (row_count, len(self.columns))


class ConfigTests(unittest.TestCase):
    def tearDown(self):
        api_module._SESSION = None

    @mock.patch.dict(os.environ, {"PANDAS_AI_MODEL": "env-model"}, clear=False)
    def test_load_config_uses_env_defaults(self):
        config = load_config()
        self.assertEqual(DEFAULT_BACKEND, config["backend"])
        self.assertEqual("not-needed", config["api_key"])
        self.assertEqual("env-model", config["model"])
        self.assertTrue(config["stream"])
        self.assertEqual(0.0, config["stream_delay"])

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_load_config_uses_default_backend_defaults(self):
        config = load_config()
        self.assertEqual(DEFAULT_BACKEND, config["backend"])
        self.assertEqual(DEFAULT_LMSTUDIO_MODEL, config["model"])
        self.assertEqual(DEFAULT_LMSTUDIO_BASE_URL, config["base_url"])
        self.assertTrue(config["stream"])
        self.assertEqual(0.0, config["stream_delay"])

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_load_config_uses_default_lmstudio_model(self):
        config = load_config(backend="lmstudio")
        self.assertEqual(DEFAULT_LMSTUDIO_MODEL, config["model"])
        self.assertEqual(DEFAULT_LMSTUDIO_BASE_URL, config["base_url"])

    def test_load_config_accepts_color(self):
        config = load_config(color="\033[102m")
        self.assertEqual("\033[102m", config["color"])

    def test_default_system_prompt_mentions_pandas_is_available(self):
        self.assertIn("Assume pandas is available.", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Omit `import pandas as pd` unless required.", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Assume an interactive Python session.", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Prefer bare expressions over `print(...)`.", DEFAULT_SYSTEM_PROMPT)


class BackendTests(unittest.TestCase):
    @mock.patch("urllib.request.urlopen")
    def test_anthropic_request_shape(self, mocked_urlopen):
        mocked_urlopen.return_value = DummyResponse({"content": [{"type": "text", "text": "result = df.head()"}]})
        backend = AnthropicBackend(
            model="claude-sonnet-4-20250514",
            api_key="secret",
            base_url="https://api.anthropic.com",
            timeout=12,
            max_tokens=321,
        )

        text = backend.generate(system_prompt="sys", user_prompt="usr")

        self.assertEqual("result = df.head()", text)
        req = mocked_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual("claude-sonnet-4-20250514", payload["model"])
        self.assertEqual(321, payload["max_tokens"])
        self.assertEqual("sys", payload["system"])
        self.assertEqual("usr", payload["messages"][0]["content"])
        self.assertEqual("secret", req.headers["X-api-key"])
        self.assertEqual("2023-06-01", req.headers["Anthropic-version"])

    @mock.patch("urllib.request.urlopen")
    def test_anthropic_http_error_is_wrapped(self, mocked_urlopen):
        mocked_urlopen.side_effect = error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=401,
            msg="unauthorized",
            hdrs={},
            fp=io.BytesIO(b'{"error":"bad key"}'),
        )
        backend = AnthropicBackend(
            model="claude-sonnet-4-20250514",
            api_key="secret",
            base_url="https://api.anthropic.com",
        )
        with self.assertRaises(BackendError):
            backend.generate(system_prompt="sys", user_prompt="usr")

    @mock.patch("urllib.request.urlopen")
    def test_openai_compat_request_shape(self, mocked_urlopen):
        mocked_urlopen.return_value = DummyResponse({"choices": [{"message": {"content": "result = df.tail()"}}]})
        backend = OpenAICompatBackend(
            model="local-model",
            api_key="token",
            base_url="http://127.0.0.1:1234/v1",
            timeout=8,
            max_tokens=88,
        )
        text = backend.generate(system_prompt="sys", user_prompt="usr")

        self.assertEqual("result = df.tail()", text)
        req = mocked_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual("local-model", payload["model"])
        self.assertEqual(88, payload["max_tokens"])
        self.assertEqual("Bearer token", req.headers["Authorization"])

    @mock.patch("urllib.request.urlopen")
    def test_openai_compat_streaming(self, mocked_urlopen):
        mocked_urlopen.return_value = DummyStreamResponse(
            [
                'data: {"choices":[{"delta":{"content":"result = "}}]}\n',
                "\n",
                'data: {"choices":[{"delta":{"content":"df.tail()"}}]}\n',
                "\n",
                "data: [DONE]\n",
                "\n",
            ]
        )
        backend = OpenAICompatBackend(
            model="local-model",
            api_key="token",
            base_url="http://127.0.0.1:1234/v1",
            timeout=8,
            max_tokens=88,
        )

        chunks = list(backend.stream_generate(system_prompt="sys", user_prompt="usr"))

        self.assertEqual(["result = ", "df.tail()"], chunks)
        req = mocked_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        self.assertTrue(payload["stream"])

    @mock.patch("urllib.request.urlopen")
    def test_anthropic_streaming(self, mocked_urlopen):
        mocked_urlopen.return_value = DummyStreamResponse(
            [
                'event: content_block_delta\n',
                'data: {"type":"content_block_delta","delta":{"text":"result = "}}\n',
                "\n",
                'data: {"type":"content_block_delta","delta":{"text":"df.head()"}}\n',
                "\n",
                'data: {"type":"message_stop"}\n',
                "\n",
            ]
        )
        backend = AnthropicBackend(
            model="claude-sonnet-4-20250514",
            api_key="secret",
            base_url="https://api.anthropic.com",
            timeout=12,
            max_tokens=321,
        )

        chunks = list(backend.stream_generate(system_prompt="sys", user_prompt="usr"))

        self.assertEqual(["result = ", "df.head()"], chunks)
        req = mocked_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        self.assertTrue(payload["stream"])


class ParsingTests(unittest.TestCase):
    def test_extract_code_from_fence(self):
        self.assertEqual("result = df.head()", extract_code("```python\nresult = df.head()\n```"))

    def test_extract_code_after_note(self):
        self.assertEqual("result = df.head()", extract_code("Here you go\nresult = df.head()"))


class PublicApiTests(unittest.TestCase):
    def tearDown(self):
        api_module._SESSION = None

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_setup_ai_creates_default_lmstudio_session(self):
        setup_ai()
        self.assertEqual("lmstudio", api_module._SESSION.backend.name)
        self.assertEqual("google/gemma-3-4b", api_module._SESSION.backend.model)
        self.assertEqual("http://192.168.40.1:1234/v1", api_module._SESSION.backend.base_url)
        self.assertTrue(api_module._SESSION.stream)

    def test_setup_ai_can_switch_to_lmstudio(self):
        setup_ai(backend="lmstudio", base_url="http://127.0.0.1:1234/v1")
        self.assertEqual("lmstudio", api_module._SESSION.backend.name)
        self.assertTrue(api_module._SESSION.stream)

    def test_setup_ai_supports_streamming_alias(self):
        setup_ai(streamming=False)
        self.assertFalse(api_module._SESSION.stream)

    def test_setup_ai_accepts_stream_handler(self):
        handler = CollectingHandler()
        setup_ai(stream=True, stream_handler=handler)
        self.assertIs(handler, api_module._SESSION.stream_handler)

    def test_setup_ai_accepts_stream_output(self):
        setup_ai(stream=True, stream_output=True)
        self.assertTrue(api_module._SESSION.stream_output)

    def test_setup_ai_accepts_stream_parse_code(self):
        setup_ai(stream=True, stream_parse_code=True)
        self.assertTrue(api_module._SESSION.stream_parse_code)

    def test_setup_ai_accepts_stream_delay(self):
        setup_ai(stream=True, stream_delay=0.25)
        self.assertEqual(0.25, api_module._SESSION.stream_delay)

    def test_setup_ai_accepts_color(self):
        setup_ai(color="\033[102m")
        self.assertEqual("\033[102m", api_module._SESSION.color)

    def test_ask_ai_accepts_single_dataframe(self):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(backend=backend, system_prompt="sys", stream=False)
        df = DummyDataFrame({"value": [1, 2, 3]})

        result = ask_ai("sum the column", dfs=df)

        self.assertEqual("result = df['value'].sum()", result)
        self.assertIn("DataFrame `df` shape=(3, 1)", backend.calls[0][1])

    def test_ask_ai_accepts_multiple_dataframes(self):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(backend=backend, system_prompt="sys", stream=False)
        df0 = DummyDataFrame({"value": [1]})
        df1 = DummyDataFrame({"other": [2]})

        ask_ai("join them", dfs=[df0, df1])

        prompt = backend.calls[0][1]
        self.assertIn("DataFrame `df0` shape=(1, 1)", prompt)
        self.assertIn("DataFrame `df1` shape=(1, 1)", prompt)

    def test_ask_ai_accepts_named_mapping(self):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(backend=backend, system_prompt="sys", stream=False)
        users = DummyDataFrame({"user_id": [1]})
        sales = DummyDataFrame({"amount": [5]})

        ask_ai("merge", dfs={"users": users, "sales": sales})

        prompt = backend.calls[0][1]
        self.assertIn("DataFrame `users` shape=(1, 1)", prompt)
        self.assertIn("DataFrame `sales` shape=(1, 1)", prompt)

    def test_ask_ai_uses_caller_df_by_default(self):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(backend=backend, system_prompt="sys", stream=False)
        df = DummyDataFrame({"value": [1, 2]})

        result = ask_ai("sum it")

        self.assertIn("result = df['value'].sum()", result)
        self.assertIn("DataFrame `df` shape=(2, 1)", backend.calls[0][1])

    def test_ask_ai_without_df_raises(self):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(backend=backend, system_prompt="sys", stream=False)
        with self.assertRaises(ConfigurationError):
            ask_ai("sum it")

    def test_ask_ai_streams_by_default(self):
        backend = FakeBackend()
        handler = CollectingHandler()
        api_module._SESSION = api_module.AISession(
            backend=backend,
            system_prompt="sys",
            stream=True,
            stream_handler=handler,
        )
        df = DummyDataFrame({"value": [1, 2, 3]})

        result = ask_ai("sum the column", dfs=df)

        self.assertEqual("result = df['value'].sum()", result)
        self.assertEqual(["```python\n", "result = df['value'].sum()\n", "```"], handler.chunks)

    def test_ask_ai_stream_parse_code_hides_fences(self):
        backend = FakeBackend()
        handler = CollectingHandler()
        api_module._SESSION = api_module.AISession(
            backend=backend,
            system_prompt="sys",
            stream=True,
            stream_parse_code=True,
            stream_handler=handler,
        )
        df = DummyDataFrame({"value": [1, 2, 3]})

        result = ask_ai("sum the column", dfs=df)

        self.assertEqual("result = df['value'].sum()", result)
        self.assertEqual(["result = df['value'].sum()"], handler.chunks)

    @mock.patch("pandas_ai.session.time.sleep")
    def test_ask_ai_stream_delay_sleeps_per_chunk(self, mocked_sleep):
        backend = FakeBackend()
        handler = CollectingHandler()
        api_module._SESSION = api_module.AISession(
            backend=backend,
            system_prompt="sys",
            stream=True,
            stream_handler=handler,
            stream_delay=0.2,
        )
        df = DummyDataFrame({"value": [1, 2, 3]})

        result = ask_ai("sum the column", dfs=df)

        self.assertEqual("result = df['value'].sum()", result)
        self.assertEqual(3, mocked_sleep.call_count)
        mocked_sleep.assert_called_with(0.2)

    def test_ask_ai_stream_false_preserves_string_result(self):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(backend=backend, system_prompt="sys", stream=True)
        df = DummyDataFrame({"value": [1, 2, 3]})

        api_module._SESSION.stream = False
        result = ask_ai("sum the column", dfs=df)

        self.assertEqual("result = df['value'].sum()", result)

    @mock.patch("sys.stdout")
    def test_ask_ai_stream_default_handler_is_silent(self, mocked_stdout):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(backend=backend, system_prompt="sys", stream=True)
        df = DummyDataFrame({"value": [1, 2, 3]})

        result = ask_ai("sum the column", dfs=df)

        self.assertEqual("result = df['value'].sum()", result)
        mocked_stdout.write.assert_not_called()
        mocked_stdout.flush.assert_not_called()

    @mock.patch("sys.stdout")
    def test_ask_ai_stream_output_writes_to_stdout(self, mocked_stdout):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(
            backend=backend,
            system_prompt="sys",
            stream=True,
            stream_output=True,
        )
        df = DummyDataFrame({"value": [1, 2, 3]})

        result = ask_ai("sum the column", dfs=df)

        self.assertEqual("result = df['value'].sum()", result)
        self.assertEqual(3, mocked_stdout.write.call_count)
        mocked_stdout.flush.assert_called()

    @mock.patch("sys.stdout")
    def test_ask_ai_stream_output_applies_color(self, mocked_stdout):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(
            backend=backend,
            system_prompt="sys",
            stream=True,
            stream_output=True,
            color="\033[102m",
        )
        df = DummyDataFrame({"value": [1, 2, 3]})

        ask_ai("sum the column", dfs=df)

        mocked_stdout.write.assert_any_call("\033[102m```python\n\033[0m")
        mocked_stdout.write.assert_any_call("\033[102mresult = df['value'].sum()\n\033[0m")
        mocked_stdout.write.assert_any_call("\033[102m```\033[0m")

    @mock.patch("sys.stdout")
    def test_ask_ai_stream_parse_code_output_applies_color_without_fences(self, mocked_stdout):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(
            backend=backend,
            system_prompt="sys",
            stream=True,
            stream_parse_code=True,
            stream_output=True,
            color="\033[102m",
        )
        df = DummyDataFrame({"value": [1, 2, 3]})

        ask_ai("sum the column", dfs=df)

        mocked_stdout.write.assert_called_once_with("\033[102mresult = df['value'].sum()\033[0m")

    def test_ask_ai_json_stream_parses_only_at_final(self):
        backend = JsonBackend()
        handler = CollectingHandler()
        api_module._SESSION = api_module.AISession(
            backend=backend,
            system_prompt="sys",
            stream=True,
            stream_handler=handler,
        )
        df = DummyDataFrame({"value": [1, 2, 3]})

        result = ask_ai("return json", dfs=df, output_format="json")

        self.assertEqual('{"ok": true, "rows": 3}', result)
        self.assertEqual(['{"ok": ', 'true, ', '"rows": 3}'], handler.chunks)


class InteractiveTests(unittest.TestCase):
    def tearDown(self):
        api_module._SESSION = None

    @mock.patch("builtins.print")
    def test_displayhook_applies_color_for_non_stream_result(self, mocked_print):
        original_displayhook = sys.displayhook
        try:
            api_module._SESSION = api_module.AISession(
                backend=FakeBackend(),
                system_prompt="sys",
                stream=False,
                color="\033[102m",
            )
            interactive_module.enable_ai_result_displayhook()

            sys.displayhook(AIResult("result = df.head()"))

            mocked_print.assert_called_once_with("\033[102mresult = df.head()\033[0m")
        finally:
            sys.displayhook = original_displayhook


if __name__ == "__main__":
    unittest.main()
