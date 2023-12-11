from abc import ABC, abstractmethod

class BaseLM(ABC):
    @property
    @abstractmethod
    def meta_prompt(self) -> str:
        pass

    @abstractmethod
    def build_prompt(self, prompt: str) -> str:
        pass

    @abstractmethod
    def prompt_completion(self, prompt: str) -> str:
        pass