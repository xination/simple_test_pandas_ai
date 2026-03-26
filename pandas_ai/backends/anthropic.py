import json
from urllib import error, request

from ..errors import BackendError, ConfigurationError
from .base import BaseBackend

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicBackend(BaseBackend):
    name = "claude"

    def __init__(self, model, api_key, base_url, timeout=30, max_tokens=1024):
        if not api_key:
            raise ConfigurationError(
                "Anthropic API key is missing. Set ANTHROPIC_API_KEY or call "
                "setup_ai(backend='claude', api_key='...'). If you are on an air-gapped "
                "server, switch to setup_ai(backend='lmstudio', base_url='http://127.0.0.1:1234/v1')."
            )
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def generate(self, system_prompt, user_prompt):
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        endpoint = "{0}/v1/messages".format(self.base_url)
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BackendError("Anthropic API request failed with status {0}: {1}".format(exc.code, detail))
        except error.URLError as exc:
            raise BackendError(
                "Unable to reach Anthropic API: {0}. If this host is air-gapped, "
                "switch to setup_ai(backend='lmstudio', base_url='http://127.0.0.1:1234/v1').".format(exc.reason)
            )

        try:
            payload = json.loads(raw)
            content = payload.get("content", [])
            text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
            text = "\n".join(part for part in text_parts if part).strip()
        except (ValueError, AttributeError) as exc:
            raise BackendError("Anthropic API returned invalid JSON: {0}".format(exc))

        if not text:
            raise BackendError("Anthropic API returned no text content.")
        return text
