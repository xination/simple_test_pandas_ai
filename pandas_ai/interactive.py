import builtins
import sys

from . import api as api_module
from .session import AIResult, _default_stream_handler

def color_stream_handler(chunk):
    print(chunk, end="", flush=True)


def enable_ai_result_displayhook():
    original_displayhook = sys.displayhook

    def displayhook(value):
        if isinstance(value, AIResult):
            session = api_module._SESSION
            has_visible_stream_output = (
                session is not None
                and session.stream
                and (session.stream_output or session.stream_handler is not _default_stream_handler)
            )
            if has_visible_stream_output:
                print("")
            else:
                print(value)
            builtins._ = value
            return
        original_displayhook(value)

    sys.displayhook = displayhook
