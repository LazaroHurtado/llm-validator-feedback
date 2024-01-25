from commands import ModelArgs
from model.gguf_model import GgufModel
from model.hf_model import HfModel

import textwrap

def format_meta_prompt(meta_prompt: str) -> str:
    return textwrap.dedent(meta_prompt).replace('\n',' ').strip()

def model_from_args(model_args: ModelArgs):
        if model_args.is_gguf:
            return GgufModel(model_args)
        return HfModel(model_args)