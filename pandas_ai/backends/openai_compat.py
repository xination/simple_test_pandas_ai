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
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        endpoint = "{0}/chat/completions".format(self.base_url)
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "authorization": "Bearer {0}".format(self.api_key),
        }
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BackendError("OpenAI-compatible request failed with status {0}: {1}".format(exc.code, detail))
        except error.URLError as exc:
            raise BackendError("Unable to reach OpenAI-compatible backend: {0}".format(exc.reason))

        try:
            payload = json.loads(raw)
            choices = payload.get("choices", [])
            text = choices[0]["message"]["content"].strip()
        except (ValueError, AttributeError, IndexError, KeyError) as exc:
            raise BackendError("OpenAI-compatible backend returned invalid JSON: {0}".format(exc))

        if not text:
            raise BackendError("OpenAI-compatible backend returned no text content.")
        return text
