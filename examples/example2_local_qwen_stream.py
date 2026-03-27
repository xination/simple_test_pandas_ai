"""Local streaming demo using a small Hugging Face model.

This example is intentionally separate from the built-in backends so the core
package stays stdlib-only. It injects a custom backend into pandas_ai and uses
`stream=True` with a visible stream handler.

Required extra packages for this example only:
    pip install pandas torch transformers huggingface_hub

Optional environment variables:
    HF_TOKEN
    PANDAS_AI_LOCAL_MODEL
    PANDAS_AI_STREAM_DELAY
"""

from pathlib import Path
from threading import Thread
import os
import sys
import time

import pandas as pd
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pandas_ai import ask_ai
from pandas_ai import api as api_module
from pandas_ai.config import DEFAULT_SYSTEM_PROMPT
from pandas_ai.session import AISession


def build_demo_dataframe():
    records = [
        {"community_area": "Lincoln Park", "list_price": 525000, "days_on_market": 18},
        {"community_area": "Lincoln Park", "list_price": 1295000, "days_on_market": 34},
        {"community_area": "Lakeview", "list_price": 289000, "days_on_market": 11},
        {"community_area": "Lakeview", "list_price": 699000, "days_on_market": 27},
        {"community_area": "Near North Side", "list_price": 615000, "days_on_market": 22},
        {"community_area": "Near North Side", "list_price": 879000, "days_on_market": 40},
        {"community_area": "Hyde Park", "list_price": 319000, "days_on_market": 15},
        {"community_area": "Hyde Park", "list_price": 845000, "days_on_market": 51},
    ]
    return pd.DataFrame.from_records(records)


class LocalTransformersBackend(object):
    name = "qwen-local"

    def __init__(self, model_name, max_new_tokens=192, hf_token=None):
        if hf_token:
            login(token=hf_token)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        )

    def generate(self, system_prompt, user_prompt):
        generated_ids, model_inputs = self._generate_ids(system_prompt, user_prompt)
        trimmed_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

    def stream_generate(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "streamer": streamer,
        }
        worker = Thread(target=self.model.generate, kwargs=generation_kwargs)
        worker.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            worker.join()

    def _generate_ids(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
        )
        return generated_ids, model_inputs


def build_stream_handler(delay_seconds):
    def handle_chunk(chunk):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return handle_chunk

GREEN = "\033[102m"
COLOREND = "\033[49m"

if __name__ == "__main__":
    model_name = os.environ.get("PANDAS_AI_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    hf_token = os.environ.get("HF_TOKEN")
    stream_delay = float(os.environ.get("PANDAS_AI_STREAM_DELAY", "0.03"))

    backend = LocalTransformersBackend(model_name=model_name, hf_token=hf_token)
    api_module._SESSION = AISession(
        backend=backend,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        stream=True,
        stream_handler=build_stream_handler(stream_delay),
    )

    df = build_demo_dataframe()

    print("Model:", model_name)
    print('Try: ask_ai("average list_price by community_area")')
    print("\n==========Streaming output:==========\n")
    result = ask_ai("average list_price by community_area", dfs=df)
    print(GREEN + "\n\nFinal  result:\n{0}".format(result) + COLOREND )
