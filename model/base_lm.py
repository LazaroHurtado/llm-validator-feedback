from commands import ModelArgs
from helpers.causal_lm import tokenize, detokenize, generate, prepare_model_config

from abc import ABC, abstractmethod

class BaseLM(ABC):

    def __init__(self, model_args: ModelArgs) -> None:
        self.name = model_args.name
        self.model_args = model_args
        self.generation_config = model_args.generation_config
        self.llm = None

        self.max_tokens = self.model_args.context_length - self.generation_config.get("max_new_tokens", 0)

    @property
    def meta_prompt(self) -> str:
        pass

    @abstractmethod
    def build_prompt(self, prompt: str) -> str:
        pass

    @abstractmethod
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def prompt_completion(self, prompt: str) -> str:
        pass

    def structure_prompt(self, prompt):
        return "".join([
            self.model_args.prompt_prefix,
            prompt,
            self.model_args.prompt_suffix])

    def load_model(self):
        if self.llm is not None:
            return
        
        self.LOGGER.debug(f"Loading model named {self.name}")
        
        if self.model_args.is_gguf:  
            from ctransformers import AutoModelForCausalLM, AutoConfig

            self.generation_config["context_length"] = self.model_args.context_length
            config = AutoConfig.from_pretrained(self.name, **self.generation_config)
            
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.name,
                model_file=self.model_args.file,
                config=config
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self.model_args.device
            model_config = prepare_model_config(self.model_args.model_config)

            tokenizer = AutoTokenizer.from_pretrained(self.name)
            
            self.llm = AutoModelForCausalLM.from_pretrained(self.name, **model_config).to(device)
            self.generate = lambda prompt : generate(self.llm,
                                                     tokenizer,
                                                     prompt,
                                                     self.generation_config)

            self.llm.tokenize = lambda prompt : tokenize(prompt, tokenizer)["input_ids"]
            self.llm.detokenize = lambda tokens : detokenize(tokens, tokenizer)