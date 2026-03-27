class BaseBackend(object):
    name = "base"

    def generate(self, system_prompt, user_prompt):
        raise NotImplementedError

    def stream_generate(self, system_prompt, user_prompt):
        raise NotImplementedError
