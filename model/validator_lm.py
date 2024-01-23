from commands import ModelArgs
from helpers.causal_lm import format_meta_prompt
from helpers.model_logger import ModelLogger
from model.base_lm import BaseLM

class ValidatorLM(BaseLM):
    LOGGER = ModelLogger("VALIDATOR")
    META_PROMPT: str = format_meta_prompt("""
        You are a validator who judges a piece of text and validates if it
        satisfies some constraints. It is extremely important
        that you are correct, there will be huge consequences if not.
        **DO NOT HALLUCINATE**. Let's think step by step to make sure we are correct.
        If the text does not satisfy a constraint then output invalid. If the text
        satisfies all the constraints then output valid.""")
    CONSTRAINTS_PROMPT: str = format_meta_prompt("Constraints: {constraints}")
    TEXT_PROMPT: str = format_meta_prompt("Text: {text}")

    def __init__(self, model_args: ModelArgs, eager_load: bool = False):
        super().__init__(model_args)

        if (eager_load):
            self.load_model()

    @property
    def meta_prompt(self) -> str:
        return self.META_PROMPT
    
    def build_prompt(self, prompt: str) -> str:
        constraints_prompt = self.CONSTRAINTS_PROMPT.format(constraints=f"\n{self.constraints}")
        text_prompt = self.TEXT_PROMPT.format(text=f"\n{prompt}")

        full_prompt = f"{self.META_PROMPT}\n\n{constraints_prompt}\n\n{text_prompt}"
        structured_prompt = self.structure_prompt(full_prompt)
        
        tokenized_prompt = self.llm.tokenize(structured_prompt)[:self.max_tokens]
        trimmed_prompt = self.llm.detokenize(tokenized_prompt)

        return trimmed_prompt

    def generate(self, prompt: str) -> str:
        self.load_model()

        input = self.build_prompt(prompt)
        output = self.prompt_completion(input)

        return output
    
    @LOGGER.log_completion
    def prompt_completion(self, prompt: str) -> str:
        if self.model_args.is_gguf:
            return self.llm(prompt)
        return self.generate(prompt)
    
    def set_constraints(self, constraints: str):
        self.constraints = constraints
