from commands import ModelArgs

from abc import ABC, abstractmethod

import torch

class Model(ABC):

    def __init__(self, model_args: ModelArgs):
        self.name = model_args.name
        self.model_args = model_args

        self.model_config = self.model_args.model_config
        dtype_str = self.model_config.get("torch_dtype", None)
        if dtype_str is not None:
            self.model_config["torch_dtype"] = getattr(torch, dtype_str)
        
        self.generation_config = model_args.generation_config
        self.generation_config["context_length"] = self.model_args.context_length

        self.load()

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def tokenize(self, prompt: str):
        pass

    @abstractmethod
    def detokenize(self, tokens: str):
        pass

    @abstractmethod
    def load(self):
        pass