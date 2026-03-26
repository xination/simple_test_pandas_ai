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
    def __init__(self, backend, system_prompt):
        self.backend = backend
        self.system_prompt = system_prompt

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
        return cls(backend=backend, system_prompt=config["system_prompt"])

    def ask_ai(self, text, dfs=None):
        df_map = normalize_dfs(dfs)
        user_prompt = build_user_prompt(text=text, df_map=df_map)
        raw_text = self.backend.generate(system_prompt=self.system_prompt, user_prompt=user_prompt)
        code = extract_code(raw_text)
        return AIResult("[{0}] {1}".format(self.backend.name, code))
