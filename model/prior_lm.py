import textwrap

from ctransformers import AutoModelForCausalLM, AutoConfig
from commands import ModelArgs
from model.base_lm import BaseLM
from helpers.model_logger import ModelLogger

class PriorLM(BaseLM):
    LOGGER = ModelLogger("PRIOR")
    PROMPT: str = textwrap.dedent("""
        You are a helpful assistant. You will summarize interesting trend
        from a government analytics report. Take a deep breath. Read through
        the government report carefully before proceed...... You need to
        summarize the insights. The summary should be placed at the beginning
        of the output. DO NOT ADD "_Summary_:" before you state summary content.
        Do not mention "THE REPORT shows..." in the beginning. Readers may not
        be aware of the report you refer to. When something is unusal, do not
        mention further investigation is needed. Also, Be specific when mentioned
        trend, increasing, decreasing, or not changing. REMOVE unclear comments,
        for example, `other sectors show sligh growth`, because this kind of
        comments add little value to the insights. Only mention insights that
        really matters.
        """)

    def __init__(self, model_args: ModelArgs, eager_load: bool = False):
        self.name = model_args.name
        self.model_args = model_args
        self.lm_config = model_args.config
        self.llm = None

        self.examples = []

        if (eager_load):
            self.load_model()

    @property
    def meta_prompt(self) -> str:
        return self.PROMPT
    
    def build_prompt(self, prompt: str) -> str:
        full_prompt = "\n\n".join(self.examples + [
            self.meta_prompt,
            prompt])
        
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

    def set_examples(self, examples: list[str]):
        self.examples = examples

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