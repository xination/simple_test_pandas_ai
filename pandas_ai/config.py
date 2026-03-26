import os

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_SYSTEM_PROMPT = (
    "You are a pandas assistant. Return only Python/pandas code that the user "
    "can copy and paste into a Python REPL. Use only the provided dataframe "
    "names. Do not add explanations unless it is a short Python comment on the "
    "first line. Prefer concise code."
)


def _coalesce(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return None


def load_config(
    backend="claude",
    model=None,
    api_key=None,
    base_url=None,
    system_prompt=None,
    timeout=30,
    max_tokens=1024,
    extra=None,
):
    env_backend = os.environ.get("PANDAS_AI_BACKEND")
    env_model = os.environ.get("PANDAS_AI_MODEL")
    env_timeout = os.environ.get("PANDAS_AI_TIMEOUT")
    env_base_url = os.environ.get("PANDAS_AI_BASE_URL")

    backend_name = _coalesce(backend, env_backend) or "claude"
    backend_name = str(backend_name).strip().lower()

    resolved_timeout = int(_coalesce(timeout, env_timeout) or 30)
    resolved_model = _coalesce(model, env_model)
    resolved_system_prompt = _coalesce(system_prompt, os.environ.get("PANDAS_AI_SYSTEM_PROMPT"))

    if backend_name == "claude":
        resolved_base_url = _coalesce(base_url, env_base_url) or DEFAULT_ANTHROPIC_BASE_URL
        resolved_model = resolved_model or DEFAULT_ANTHROPIC_MODEL
        resolved_api_key = _coalesce(api_key, os.environ.get("ANTHROPIC_API_KEY"))
    elif backend_name == "lmstudio":
        resolved_base_url = _coalesce(base_url, env_base_url) or DEFAULT_LMSTUDIO_BASE_URL
        resolved_model = resolved_model or "local-model"
        resolved_api_key = _coalesce(api_key, os.environ.get("OPENAI_API_KEY"), "not-needed")
    else:
        resolved_base_url = _coalesce(base_url, env_base_url)
        resolved_api_key = api_key

    return {
        "backend": backend_name,
        "model": resolved_model,
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
        "timeout": resolved_timeout,
        "max_tokens": int(max_tokens),
        "system_prompt": resolved_system_prompt or DEFAULT_SYSTEM_PROMPT,
        "extra": extra or {},
    }
