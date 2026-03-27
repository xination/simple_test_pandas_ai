import json
from urllib import error, request

from ..errors import BackendError
from .base import BaseBackend


class OpenAICompatBackend(BaseBackend):
    name = "lmstudio"

    def __init__(self, model, api_key, base_url, timeout=30, max_tokens=1024):
        self.model = model
        self.api_key = api_key or "not-needed"
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def generate(self, system_prompt, user_prompt):
        raw = self._request(system_prompt=system_prompt, user_prompt=user_prompt, stream=False)
        return self._parse_response(raw)

    def stream_generate(self, system_prompt, user_prompt):
        response = self._request(system_prompt=system_prompt, user_prompt=user_prompt, stream=True)
        for payload in self._iter_sse_payloads(response):
            if payload == "[DONE]":
                break
            try:
                body = json.loads(payload)
                choices = body.get("choices", [])
                delta = choices[0].get("delta", {})
                text = delta.get("content", "")
            except (ValueError, AttributeError, IndexError, KeyError) as exc:
                raise BackendError("OpenAI-compatible streaming payload was invalid: {0}".format(exc))
            if text:
                yield text

    def _request(self, system_prompt, user_prompt, stream):
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if stream:
            payload["stream"] = True
        endpoint = "{0}/chat/completions".format(self.base_url)
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "authorization": "Bearer {0}".format(self.api_key),
        }
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            return request.urlopen(req, timeout=self.timeout)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BackendError("OpenAI-compatible request failed with status {0}: {1}".format(exc.code, detail))
        except error.URLError as exc:
            raise BackendError("Unable to reach OpenAI-compatible backend: {0}".format(exc.reason))

    def _parse_response(self, response):
        try:
            with response:
                raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            choices = payload.get("choices", [])
            text = choices[0]["message"]["content"].strip()
        except (ValueError, AttributeError, IndexError, KeyError) as exc:
            raise BackendError("OpenAI-compatible backend returned invalid JSON: {0}".format(exc))

        if not text:
            raise BackendError("OpenAI-compatible backend returned no text content.")
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
