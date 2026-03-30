import json
import sys
import time

from .backends.anthropic import AnthropicBackend
from .backends.openai_compat import OpenAICompatBackend
from .errors import ConfigurationError
from .parsing import extract_code
from .prompting import build_user_prompt
from .schema import normalize_dfs

class AIResult(str):
    def __repr__(self):
        return str(self)


class AISession(object):
    def __init__(
        self,
        backend,
        system_prompt,
        stream=True,
        stream_parse_code=False,
        stream_output=False,
        stream_handler=None,
        stream_delay=0.0,
        color=None,
    ):
        self.backend = backend
        self.system_prompt = system_prompt
        self.color = color
        self.stream = stream
        self.stream_parse_code = stream_parse_code
        self.stream_output = stream_output
        self.stream_handler = _resolve_stream_handler(
            stream_output=stream_output,
            stream_handler=stream_handler,
            color=color,
        )
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
            color=config.get("color"),
            stream=config["stream"],
            stream_parse_code=config.get("stream_parse_code", False),
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
        stream_parser = _StreamCodeParser(enabled=self.stream_parse_code and output_format == "text")
        for chunk in self.backend.stream_generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        ):
            if chunk:
                chunks.append(chunk)
                for visible_chunk in stream_parser.feed(chunk):
                    self.stream_handler(visible_chunk)
                if self.stream_delay > 0:
                    time.sleep(self.stream_delay)

        for visible_chunk in stream_parser.finish():
            self.stream_handler(visible_chunk)

        raw_text = "".join(chunks)
        return self._format_final_result(raw_text=raw_text, output_format=output_format)

    def _format_final_result(self, raw_text, output_format):
        if output_format == "json":
            json.loads(raw_text)
            return raw_text
        if output_format != "text":
            raise ConfigurationError("Unsupported output_format `{0}`. Supported values: text, json.".format(output_format))
        code = extract_code(raw_text)
        return AIResult(code)


def _default_stream_handler(chunk):
    return None


class _StreamCodeParser(object):
    def __init__(self, enabled=False):
        self.enabled = enabled
        self._buffer = ""
        self._started = False
        self._ended = False

    def feed(self, chunk):
        if not self.enabled or self._ended:
            return [chunk]

        self._buffer += chunk
        visible_chunks = []

        if not self._started:
            newline_index = self._buffer.find("\n")
            if newline_index == -1:
                return []
            first_line = self._buffer[: newline_index + 1]
            if first_line.startswith("```"):
                self._buffer = self._buffer[newline_index + 1 :]
                self._started = True
            else:
                self._started = True

        if self._buffer.endswith("```"):
            content = self._buffer[:-3]
            if content.endswith("\n"):
                content = content[:-1]
            if content:
                visible_chunks.append(content)
            self._buffer = ""
            self._ended = True
            return visible_chunks

        if self._buffer.endswith("\n"):
            content = self._buffer[:-1]
            if content:
                visible_chunks.append(content)
            self._buffer = "\n"
            return visible_chunks

        if self._buffer:
            visible_chunks.append(self._buffer)
            self._buffer = ""
        return visible_chunks

    def finish(self):
        if not self.enabled:
            return []
        if self._ended or not self._buffer:
            return []
        return [self._buffer]


def _apply_color(text, color=None):
    if not color:
        return text
    return "{0}{1}\033[0m".format(color, text)


def _stdout_stream_handler(chunk, color=None):
    sys.stdout.write(_apply_color(chunk, color=color))
    sys.stdout.flush()


def _resolve_stream_handler(stream_output, stream_handler, color=None):
    if stream_handler is not None:
        return stream_handler
    if stream_output:
        return lambda chunk: _stdout_stream_handler(chunk, color=color)
    return _default_stream_handler
