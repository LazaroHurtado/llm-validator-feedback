from commands import ModelArgs
from helpers.causal_lm import format_meta_prompt
from helpers.model_logger import ModelLogger
from model.base_lm import BaseLM

class ExtractorLM(BaseLM):
    LOGGER = ModelLogger("EXTRACTOR")
    META_PROMPT: str = format_meta_prompt("""
        You will take in a user's request and extract all the necessary
        constraints the user is asking for. Be precise and detailed. Only
        output the constraints that are being asked for. Do not include
        "the user" when outputting the constraints. For the following
        text what are the constraints?""")

    def __init__(self, model_args: ModelArgs, eager_load: bool = False):
        super().__init__(model_args)

        if (eager_load):
            self.load_model()

    @property
    def meta_prompt(self) -> str:
        return self.META_PROMPT
    
    def build_prompt(self, prompt: str) -> str:
        full_prompt = f"{self.META_PROMPT}\n\n{prompt}"
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
        generation_warming = "Constraints:\n1."
        new_prompt = f"{prompt}\n\n{generation_warming}"
        if self.model_args.is_gguf:
            return generation_warming+self.llm(new_prompt)
        return generation_warming+self.generate(new_prompt)
