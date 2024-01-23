from dataclasses import dataclass, field, replace
from jsonargparse import CLI
from typing import Optional, Dict

import yaml

@dataclass
class ModelArgs:
    name: str
    file: Optional[str] = None
    context_length: int = 2048
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    is_gguf: bool = False
    device: str = "cpu"
    generation_config: Dict[str, any] = field(default_factory=dict)
    model_config: Dict[str, any] = field(default_factory=dict)

@dataclass
class Args:
    model: Optional[ModelArgs]
    dataset: Optional[str]
    prompt: Optional[str]
    from_yml: Optional[str]

    def __post_init__(self):
        if self.model is not None:
            self.model = ModelArgs(**self.model)

def get_args_from_yml(filename: str) -> dict:
    with open(filename, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
            exit(1)

def parse_args():
    args = CLI(Args, as_positional=False)

    if (args.from_yml is not None):
        args = replace(args, **get_args_from_yml(args.from_yml))

    return args
