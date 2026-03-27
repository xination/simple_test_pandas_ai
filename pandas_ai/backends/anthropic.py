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
        response = self._request(system_prompt=system_prompt, user_prompt=user_prompt, stream=False)
        return self._parse_response(response)

    def stream_generate(self, system_prompt, user_prompt):
        response = self._request(system_prompt=system_prompt, user_prompt=user_prompt, stream=True)
        for payload in self._iter_sse_payloads(response):
            try:
                body = json.loads(payload)
                event_type = body.get("type")
                if event_type == "content_block_delta":
                    text = body.get("delta", {}).get("text", "")
                else:
                    text = ""
            except (ValueError, AttributeError) as exc:
                raise BackendError("Anthropic streaming payload was invalid: {0}".format(exc))
            if text:
                yield text

    def _request(self, system_prompt, user_prompt, stream):
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if stream:
            payload["stream"] = True
        endpoint = "{0}/v1/messages".format(self.base_url)
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            return request.urlopen(req, timeout=self.timeout)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BackendError("Anthropic API request failed with status {0}: {1}".format(exc.code, detail))
        except error.URLError as exc:
            raise BackendError(
                "Unable to reach Anthropic API: {0}. If this host is air-gapped, "
                "switch to setup_ai(backend='lmstudio', base_url='http://127.0.0.1:1234/v1').".format(exc.reason)
            )

    def _parse_response(self, response):
        try:
            with response:
                raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            content = payload.get("content", [])
            text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
            text = "\n".join(part for part in text_parts if part).strip()
        except (ValueError, AttributeError) as exc:
            raise BackendError("Anthropic API returned invalid JSON: {0}".format(exc))

        if not text:
            raise BackendError("Anthropic API returned no text content.")
        return text

    def _iter_sse_payloads(self, response):
        with response:
            data_lines = []
            while True:
                line = response.readline()
                if not line:
                    break
                text = line.decode("utf-8").strip()
                if not text:
                    if data_lines:
                        yield "\n".join(data_lines)
                        data_lines = []
                    continue
                if text.startswith("data:"):
                    data_lines.append(text[5:].strip())
