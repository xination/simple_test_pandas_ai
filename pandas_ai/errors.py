class PandasAIError(Exception):
    pass


class ConfigurationError(PandasAIError):
    pass


class BackendError(PandasAIError):
    pass


class ResponseParseError(PandasAIError):
    pass
