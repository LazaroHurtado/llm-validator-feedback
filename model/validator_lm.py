import logging

from ctransformers import AutoModelForCausalLM
from commands import ModelArgs
from model.base_lm import BaseLM

class ValidatorLM(BaseLM):
    LOGGER = logging.getLogger("[VALIDATOR]")
    PROMPT: str = """
        You are a validator who judges a piece of text and validates if it \
        satisfies some constraints. If the text does not satisfy a constraint \
        then you must output that the text is invalid. It is extremely important \
        that you are correct, there will be huge consequences if not. \
        **DO NOT HALLUCINATE**. Let's think step by step to make sure we are correct. \
        
        Constraints:
        1. Is your answer clear and precise?
        2. Is your answer within 200 words?
        3. Did you make sure the summary part is put at the beginning?
        4. Did you remove '_Summary_:'?

        Text:
    """

    def __init__(self, args: ModelArgs, eager_load: bool = False):
        self.name = args.name
        self.args = args
        self.llm = None

        if (eager_load):
            self.load_model()

    @property
    def meta_prompt(self) -> str:
        return self.PROMPT
    
    #TODO: add few-shot support
    def build_prompt(self, prompt: str) -> str:
        return f"${self.meta_prompt}\n${prompt}"

    def prompt_completion(self, prompt: str) -> str:
        self.load_model()

        input = self.build_prompt(prompt)
        output = self.llm(input)

        return output

    def load_model(self):
        if self.llm is not None:
            return
        
        self.LOGGER.debug(f"Loading model named {self.name}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.name,
            model_file=self.args.file,
            model_type=self.args.type
        )