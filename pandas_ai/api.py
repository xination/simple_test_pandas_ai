from .config import DEFAULT_BACKEND, load_config
from .session import AISession

_SESSION = None


def setup_ai(
    backend=DEFAULT_BACKEND,
    model=None,
    api_key=None,
    base_url=None,
    system_prompt=None,
    color=None,
    stream=True,
    stream_parse_code=False,
    stream_output=False,
    stream_delay=0.0,
    timeout=30,
    max_tokens=1024,
    **kwargs
):
    global _SESSION
    stream = kwargs.pop("streamming", stream)
    config = load_config(
        backend=backend,
        model=model,
        api_key=api_key,
        base_url=base_url,
        system_prompt=system_prompt,
        color=color,
        stream=stream,
        stream_parse_code=stream_parse_code,
        stream_output=stream_output,
        stream_delay=stream_delay,
        timeout=timeout,
        max_tokens=max_tokens,
        extra=kwargs,
    )
    _SESSION = AISession.from_config(config)


def ask_ai(text, dfs=None, output_format="text"):
    global _SESSION
    if _SESSION is None:
        setup_ai()
    return _SESSION.ask_ai(text=text, dfs=dfs, output_format=output_format)
