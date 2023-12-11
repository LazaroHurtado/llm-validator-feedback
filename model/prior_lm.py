import logging

from ctransformers import AutoModelForCausalLM
from commands import ModelArgs
from model.base_lm import BaseLM

class PriorLM(BaseLM):
    LOGGER = logging.getLogger("[PRIOR]")
    PROMPT: str = """
        You are a helpful assistant. You will summarize interesting trend \
        from a government analytics report. Take a deep breath. Read through \
        the government report carefully before proceed...... You need to \
        summarize the insights. The summary should be placed at the beginning \
        of the output. DO NOT ADD "_Summary_:" before you state summary content. \
        Do not mention "THE REPORT shows..." in the beginning. Readers may not \
        be aware of the report you refer to. When something is unusal, do not \
        mention further investigation is needed. Also, Be specific when mentioned \
        trend, increasing, decreasing, or not changing. REMOVE unclear comments, \
        for example, `other sectors show sligh growth`, because this kind of \
        comments add little value to the insights. Only mention insights that \
        really matters.
    """

    def __init__(self, model_args: ModelArgs, eager_load: bool = False):
        self.name = model_args.name
        self.model_args = model_args
        self.llm = None

        if (eager_load):
            self.load_model()

    @property
    def meta_prompt(self) -> str:
        return self.PROMPT
    
    def build_prompt(self, prompt: str) -> str:
        return f"${self.meta_prompt}\n\n${prompt}"

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
            model_file=self.model_args.file,
            model_type=self.model_args.type
        )