import yaml

from dataclasses import dataclass
from jsonargparse import CLI
from typing import Optional

@dataclass
class ModelArgs:
    name: str
    file: Optional[str]
    type: Optional[str]

@dataclass
class Args:
    model: Optional[ModelArgs]
    from_yml: Optional[str]
    prompt: str = "Hello there!"

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
        return get_args_from_yml(args.from_yml)
    return vars(args)
