from commands import ModelArgs
from model.model import Model

from ctransformers import AutoModelForCausalLM, AutoConfig

class GgufModel(Model):

    def __init__(self, model_args: ModelArgs):
        super().__init__(model_args)

    def generate(self, prompt: str):
        return self.model(prompt)
    
    def tokenize(self, prompt: str):
        return self.model.tokenize(prompt)
    
    def detokenize(self, tokens: str):
        return self.model.detokenize(tokens)
    
    def load(self):
        config = AutoConfig.from_pretrained(self.name, **self.generation_config)    
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            model_file=self.model_args.file,
            config=config
        )