from .schema import build_schema_text


def build_user_prompt(text, df_map):
    schema_text = build_schema_text(df_map)
    dataframe_names = ", ".join(df_map.keys())
    return (
        "You are working in a Python REPL.\n"
        "Available pandas dataframes:\n"
        "{schema}\n\n"
        "Rules:\n"
        "- Only reference these dataframe names: {names}\n"
        "- Return Python/pandas code only\n"
        "- Do not load files or create fake data\n"
        "- Prefer code that can be pasted directly into the current REPL\n"
        "- If needed, the first line may be a short Python comment\n\n"
        "User request:\n"
        "{text}\n"
    ).format(schema=schema_text, names=dataframe_names, text=text.strip())
