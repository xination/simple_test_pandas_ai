import inspect
import keyword
import re

from .errors import ConfigurationError

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised indirectly in minimal environments
    pd = None


def _is_dataframe(value):
    if pd is not None and isinstance(value, pd.DataFrame):
        return True
    required_attrs = ("columns", "dtypes", "shape")
    return all(hasattr(value, attr) for attr in required_attrs)


def _sanitize_name(name, fallback):
    if not name:
        return fallback
    candidate = re.sub(r"\W+", "_", str(name)).strip("_")
    if not candidate or candidate[0].isdigit() or keyword.iskeyword(candidate):
        return fallback
    return candidate


def _caller_df():
    frame = inspect.currentframe()
    try:
        caller = frame.f_back
        while caller is not None:
            if "df" in caller.f_locals and _is_dataframe(caller.f_locals["df"]):
                return caller.f_locals["df"]
            if "df" in caller.f_globals and _is_dataframe(caller.f_globals["df"]):
                return caller.f_globals["df"]
            caller = caller.f_back
        return None
    finally:
        del frame


def normalize_dfs(dfs):
    if dfs is None:
        df = _caller_df()
        if df is None:
            raise ConfigurationError(
                "No dataframe context found. Pass `dfs=` explicitly or define a "
                "`df` dataframe before calling ask_ai()."
            )
        return {"df": df}

    if _is_dataframe(dfs):
        return {"df": dfs}

    if isinstance(dfs, (list, tuple)):
        normalized = {}
        for index, value in enumerate(dfs):
            if not _is_dataframe(value):
                raise ConfigurationError("All entries in `dfs` must be pandas DataFrame objects.")
            normalized["df{0}".format(index)] = value
        return normalized

    if isinstance(dfs, dict):
        normalized = {}
        for index, (name, value) in enumerate(dfs.items()):
            if not _is_dataframe(value):
                raise ConfigurationError("All values in `dfs` mapping must be pandas DataFrame objects.")
            fallback = "df{0}".format(index)
            normalized[_sanitize_name(name, fallback)] = value
        return normalized

    raise ConfigurationError(
        "`dfs` must be None, a pandas DataFrame, a list/tuple of DataFrames, or a mapping of names to DataFrames."
    )


def build_schema_text(df_map):
    sections = []
    for name, df in df_map.items():
        column_lines = []
        for column in df.columns:
            dtype = str(df.dtypes[column])
            column_lines.append("- {0}: {1}".format(column, dtype))
        columns = "\n".join(column_lines) if column_lines else "- <no columns>"
        sections.append(
            "DataFrame `{name}` shape={shape}\nColumns:\n{columns}".format(
                name=name,
                shape=df.shape,
                columns=columns,
            )
        )
    return "\n\n".join(sections)
