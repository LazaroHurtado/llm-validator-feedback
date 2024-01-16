import yaml

from dataclasses import dataclass, replace
from jsonargparse import CLI
from typing import Optional

@dataclass
class ModelConfig:
    context_length: Optional[int]
    batch_size: Optional[int]
    threads: Optional[int]

@dataclass
class ModelArgs:
    name: str
    file: Optional[str]
    type: Optional[str]
    config: Optional[ModelConfig]

@dataclass
class Args:
    model: Optional[ModelArgs]
    dataset: Optional[str]
    prompt: Optional[str]
    from_yml: Optional[str]

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
    
    if (args.model is not None):
        args.model = ModelArgs(**args.model)
    
    return args
