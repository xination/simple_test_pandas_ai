import io
import json
import os
import unittest
from unittest import mock
from urllib import error

from pandas_ai import ask_ai, setup_ai
from pandas_ai import api as api_module
from pandas_ai.backends.anthropic import AnthropicBackend
from pandas_ai.backends.openai_compat import OpenAICompatBackend
from pandas_ai.config import DEFAULT_ANTHROPIC_MODEL, load_config
from pandas_ai.errors import BackendError, ConfigurationError
from pandas_ai.parsing import extract_code


class DummyResponse(object):
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeBackend(object):
    name = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return "```python\nresult = df['value'].sum()\n```"


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

    @mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key", "PANDAS_AI_MODEL": "env-model"}, clear=False)
    def test_load_config_uses_env_defaults(self):
        config = load_config()
        self.assertEqual("claude", config["backend"])
        self.assertEqual("env-key", config["api_key"])
        self.assertEqual("env-model", config["model"])

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_load_config_uses_default_claude_model(self):
        config = load_config()
        self.assertEqual(DEFAULT_ANTHROPIC_MODEL, config["model"])


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


class ParsingTests(unittest.TestCase):
    def test_extract_code_from_fence(self):
        self.assertEqual("result = df.head()", extract_code("```python\nresult = df.head()\n```"))

    def test_extract_code_after_note(self):
        self.assertEqual("result = df.head()", extract_code("Here you go\nresult = df.head()"))


class PublicApiTests(unittest.TestCase):
    def tearDown(self):
        api_module._SESSION = None

    @mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "abc"}, clear=True)
    def test_setup_ai_creates_default_claude_session(self):
        setup_ai()
        self.assertEqual("claude", api_module._SESSION.backend.name)

    def test_setup_ai_can_switch_to_lmstudio(self):
        setup_ai(backend="lmstudio", base_url="http://127.0.0.1:1234/v1")
        self.assertEqual("lmstudio", api_module._SESSION.backend.name)

    def test_ask_ai_accepts_single_dataframe(self):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(backend=backend, system_prompt="sys")
        df = DummyDataFrame({"value": [1, 2, 3]})

        result = ask_ai("sum the column", dfs=df)

        self.assertIn("[fake]", result)
        self.assertIn("result = df['value'].sum()", result)
        self.assertIn("DataFrame `df` shape=(3, 1)", backend.calls[0][1])

    def test_ask_ai_accepts_multiple_dataframes(self):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(backend=backend, system_prompt="sys")
        df0 = DummyDataFrame({"value": [1]})
        df1 = DummyDataFrame({"other": [2]})

        ask_ai("join them", dfs=[df0, df1])

        prompt = backend.calls[0][1]
        self.assertIn("DataFrame `df0` shape=(1, 1)", prompt)
        self.assertIn("DataFrame `df1` shape=(1, 1)", prompt)

    def test_ask_ai_accepts_named_mapping(self):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(backend=backend, system_prompt="sys")
        users = DummyDataFrame({"user_id": [1]})
        sales = DummyDataFrame({"amount": [5]})

        ask_ai("merge", dfs={"users": users, "sales": sales})

        prompt = backend.calls[0][1]
        self.assertIn("DataFrame `users` shape=(1, 1)", prompt)
        self.assertIn("DataFrame `sales` shape=(1, 1)", prompt)

    def test_ask_ai_uses_caller_df_by_default(self):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(backend=backend, system_prompt="sys")
        df = DummyDataFrame({"value": [1, 2]})

        result = ask_ai("sum it")

        self.assertIn("result = df['value'].sum()", result)
        self.assertIn("DataFrame `df` shape=(2, 1)", backend.calls[0][1])

    def test_ask_ai_without_df_raises(self):
        backend = FakeBackend()
        api_module._SESSION = api_module.AISession(backend=backend, system_prompt="sys")
        with self.assertRaises(ConfigurationError):
            ask_ai("sum it")


if __name__ == "__main__":
    unittest.main()
