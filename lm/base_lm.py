from model.model import Model

from abc import ABC, abstractmethod

class BaseLM(ABC):

    def __init__(self, model: Model) -> None:
        self.model = model
        self.prompt_prefix = model.model_args.prompt_prefix
        self.prompt_suffix = model.model_args.prompt_suffix

        self.max_tokens = model.model_args.context_length - model.generation_config.get("max_new_tokens", 0)

    @property
    def meta_prompt(self) -> str:
        pass

    @abstractmethod
    def build_prompt(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def prompt_completion(self, prompt: str) -> str:
        pass

    def structure_prompt(self, prompt):
        return "".join([
            self.prompt_prefix,
            prompt,
            self.prompt_suffix])