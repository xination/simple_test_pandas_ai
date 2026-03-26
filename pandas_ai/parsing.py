import re

from .errors import ResponseParseError

CODE_PATTERNS = [
    r"```python\s*(.*?)```",
    r"```py\s*(.*?)```",
    r"```\s*(.*?)```",
]


def extract_code(text):
    if text is None:
        raise ResponseParseError("Model returned an empty response.")

    cleaned = text.strip()
    if not cleaned:
        raise ResponseParseError("Model returned an empty response.")

    for pattern in CODE_PATTERNS:
        match = re.search(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

    lines = cleaned.splitlines()
    if len(lines) > 1 and not _looks_like_code(lines[0]) and _looks_like_code("\n".join(lines[1:])):
        cleaned = "\n".join(lines[1:]).strip()

    if not cleaned:
        raise ResponseParseError("Unable to extract code from the model response.")
    return cleaned


def _looks_like_code(text):
    tokens = ("df", "pd.", "=", ".loc[", ".groupby(", ".merge(", "print(", "#")
    return any(token in text for token in tokens)
