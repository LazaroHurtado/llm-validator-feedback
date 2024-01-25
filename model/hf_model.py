from commands import ModelArgs
from model.model import Model

from transformers import AutoModelForCausalLM, AutoTokenizer

class HfModel(Model):

    def __init__(self, model_args: ModelArgs):
        super().__init__(model_args)

    def generate(self, prompt: str) -> str:
        tokenized_prompt = self.tokenize(prompt, tokens_only=False)
        inputs = tokenized_prompt | self.generation_config
        outputs = self.model.generate(**inputs)

        return self.detokenize(outputs)
    
    def tokenize(self, prompt: str, tokens_only=True):
        return self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    
    def detokenize(self, tokens: str):
        return self.tokenizer.batch_decode(tokens)[0]
    
    def load(self):
        device = self.model_args.device

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, **self.model_config).to(device)