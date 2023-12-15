import textwrap
from ctransformers import AutoModelForCausalLM, AutoConfig
from commands import ModelArgs
from model.base_lm import BaseLM
from helpers.model_logger import ModelLogger

class ValidatorLM(BaseLM):
    LOGGER = ModelLogger("VALIDATOR")
    PROMPT: str = textwrap.dedent("""
        You are a validator who judges a piece of text and validates if it
        satisfies some constraints. It is extremely important
        that you are correct, there will be huge consequences if not.
        **DO NOT HALLUCINATE**. Let's think step by step to make sure we are correct.
        If the text does not satisfy a constraint then output invalid. If the text
        satisfies all the constraints then output valid.
        
        Constraints:
        1. Is your answer clear and precise?
        2. Is your answer within 200 words?
        3. Did you make sure the summary part is put at the beginning?
        4. Did you remove '_Summary_:'?

        Text:
        """)

    def __init__(self, model_args: ModelArgs, eager_load: bool = False):
        self.name = model_args.name
        self.model_args = model_args
        self.lm_config = model_args.config
        self.llm = None

        if (eager_load):
            self.load_model()

    @property
    def meta_prompt(self) -> str:
        return self.PROMPT
    
    def build_prompt(self, prompt: str) -> str:
        full_prompt = f"{self.meta_prompt}\n{prompt}"
        
        model_name = self.name.lower()
        model_type = self.model_args.type.lower()
        if model_type == "mistral" and "instruct" in model_name:
            full_prompt = f"<s>[INST]{full_prompt}[/INST]"
        
        max_tokens = self.lm_config["context_length"] - self.lm_config["max_new_tokens"]
        tokenized_prompt = self.llm.tokenize(full_prompt)[:max_tokens]
        trimmed_prompt = self.llm.detokenize(tokenized_prompt)

        return trimmed_prompt

    def generate(self, prompt: str) -> str:
        self.load_model()

        input = self.build_prompt(prompt)
        output = self.prompt_completion(input)

        return output
    
    @LOGGER.log_completion
    def prompt_completion(self, prompt: str) -> str:
        return self.llm(prompt)

    def load_model(self):
        if self.llm is not None:
            return
        
        self.LOGGER.debug(f"Loading model named {self.name}")
        
        config = AutoConfig.from_pretrained(self.name, **self.lm_config)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.name,
            model_file=self.model_args.file,
            model_type=self.model_args.type,
            config=config
        )