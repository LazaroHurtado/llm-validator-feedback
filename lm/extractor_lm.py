from helpers.utils import format_meta_prompt
from helpers.model_logger import ModelLogger
from lm.base_lm import BaseLM
from model.model import Model

class ExtractorLM(BaseLM):
    LOGGER = ModelLogger("EXTRACTOR")
    META_PROMPT: str = format_meta_prompt("""
        You will take in a user's request and extract all the necessary
        constraints the user is asking for. Be precise and detailed. Only
        output the constraints that are being asked for. Do not include
        "the user" when outputting the constraints. For the following
        text what are the constraints?""")

    def __init__(self, model: Model):
        super().__init__(model)

    @property
    def meta_prompt(self) -> str:
        return self.META_PROMPT
    
    def build_prompt(self, prompt: str) -> str:
        full_prompt = f"{self.META_PROMPT}\n\n{prompt}"
        structured_prompt = self.structure_prompt(full_prompt)
        
        tokenized_prompt = self.model.tokenize(structured_prompt)[:self.max_tokens]
        trimmed_prompt = self.model.detokenize(tokenized_prompt)

        return trimmed_prompt

    def generate(self, prompt: str) -> str:
        input = self.build_prompt(prompt)
        output = self.prompt_completion(input)

        return output
    
    @LOGGER.log_completion
    def prompt_completion(self, prompt: str) -> str:
        generation_warming = "Constraints:\n1."
        new_prompt = f"{prompt}\n\n{generation_warming}"

        return "1."+self.model.generate(new_prompt)
