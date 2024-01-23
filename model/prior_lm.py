from commands import ModelArgs
from helpers.causal_lm import format_meta_prompt
from helpers.model_logger import ModelLogger
from model.base_lm import BaseLM

class PriorLM(BaseLM):
    LOGGER = ModelLogger("PRIOR")
    META_PROMPT: str = format_meta_prompt("""
        You are a helpful assistant. You will summarize interesting trends
        from analytical reports. Take a deep breath. Read through the report
        carefully before proceeding. You need to summarize the insights. The
        summary should be placed at the beginning of the output. DO NOT ADD
        "_Summary_:" before you state summary content. Do not mention
        "THE REPORT shows..." in the beginning. Readers may not be aware of
        the report you refer to. When something is unusal, do not mention further
        investigation is needed. Also, mention when a trend is increasing, decreasing,
        or there is no change. REMOVE unclear comments because this kind of
        comments add little value to the insights. Only mention insights that
        really matters.""")
    FEW_SHOT_PROMPT: str = format_meta_prompt("""
        Here are some examples of invalid summarizations that you should not generate: {examples}""")
    TEXT_PROMPT: str = format_meta_prompt("""
        Now summarize the following text with what you have learned: {text}""")

    def __init__(self, model_args: ModelArgs, eager_load: bool = False):
        super().__init__(model_args)

        self.examples = []

        if (eager_load):
            self.load_model()

    @property
    def meta_prompt(self) -> str:
        return self.META_PROMPT
    
    def build_prompt(self, prompt: str) -> str:
        examples = "\n\n".join(self.examples)
        few_shot_prompt = self.FEW_SHOT_PROMPT.format(examples=f"\n{examples}")
        text_prompt = self.TEXT_PROMPT.format(text=f"\n{prompt}")

        full_prompt = f"{self.META_PROMPT}\n\n{few_shot_prompt}\n\n{text_prompt}"
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

    def set_examples(self, examples: list[str]):
        self.examples = examples
