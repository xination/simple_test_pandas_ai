import json
import sys
import time

from .backends.anthropic import AnthropicBackend
from .backends.openai_compat import OpenAICompatBackend
from .errors import ConfigurationError
from .parsing import extract_code
from .prompting import build_user_prompt
from .schema import normalize_dfs

GREEN = "\033[102m"
COLOREND = "\033[49m"

class AIResult(str):
    def __repr__(self):
        return str(self)


class AISession(object):
    def __init__(self, backend, system_prompt, stream=True, stream_output=False, stream_handler=None, stream_delay=0.0):
        self.backend = backend
        self.system_prompt = system_prompt
        self.stream = stream
        self.stream_output = stream_output
        self.stream_handler = _resolve_stream_handler(stream_output=stream_output, stream_handler=stream_handler)
        self.stream_delay = float(stream_delay or 0.0)

    @classmethod
    def from_config(cls, config):
        backend_name = config["backend"]
        if backend_name == "claude":
            backend = AnthropicBackend(
                model=config["model"],
                api_key=config["api_key"],
                base_url=config["base_url"],
                timeout=config["timeout"],
                max_tokens=config["max_tokens"],
            )
        elif backend_name == "lmstudio":
            backend = OpenAICompatBackend(
                model=config["model"],
                api_key=config["api_key"],
                base_url=config["base_url"],
                timeout=config["timeout"],
                max_tokens=config["max_tokens"],
            )
        else:
            raise ConfigurationError(
                "Unsupported backend `{0}`. Supported backends: claude, lmstudio.".format(backend_name)
            )
        return cls(
            backend=backend,
            system_prompt=config["system_prompt"],
            stream=config["stream"],
            stream_output=config.get("stream_output", False),
            stream_handler=config.get("stream_handler"),
            stream_delay=config.get("stream_delay", 0.0),
        )

    def ask_ai(self, text, dfs=None, output_format="text"):
        df_map = normalize_dfs(dfs)
        user_prompt = build_user_prompt(text=text, df_map=df_map)
        if self.stream:
            return self._stream_ai(user_prompt=user_prompt, output_format=output_format)

        raw_text = self.backend.generate(system_prompt=self.system_prompt, user_prompt=user_prompt)
        return self._format_final_result(raw_text=raw_text, output_format=output_format)

    def _stream_ai(self, user_prompt, output_format):
        chunks = []
        for chunk in self.backend.stream_generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        ):
            if chunk:
                chunks.append(chunk)
                self.stream_handler(chunk)
                if self.stream_delay > 0:
                    time.sleep(self.stream_delay)

        raw_text = "".join(chunks)
        return self._format_final_result(raw_text=raw_text, output_format=output_format)

    def _format_final_result(self, raw_text, output_format):
        if output_format == "json":
            json.loads(raw_text)
            return raw_text
        if output_format != "text":
            raise ConfigurationError("Unsupported output_format `{0}`. Supported values: text, json.".format(output_format))
        code = extract_code(raw_text)
        return AIResult("{0}{1}{2}".format(GREEN, code, COLOREND))


def _default_stream_handler(chunk):
    return None


def _stdout_stream_handler(chunk):
    sys.stdout.write(chunk)
    sys.stdout.flush()


def _resolve_stream_handler(stream_output, stream_handler):
    if stream_handler is not None:
        return stream_handler
    if stream_output:
        return _stdout_stream_handler
    return _default_stream_handler
