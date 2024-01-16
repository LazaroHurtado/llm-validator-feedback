from model.base_lm import BaseLM
from helpers.model_logger import ModelLogger

class Coordinator():
    LOGGER = ModelLogger("COORDINATOR")

    def __init__(self, prior: BaseLM, validator: BaseLM, retries: int = 3):
        self.prior = prior
        self.validator = validator

        self.retries = retries

    @LOGGER.log_completion
    def call(self, prompt: str) -> str:
        for attempt in range(self.retries):
            self.LOGGER.info(f"Attempt {attempt} of {self.retries}")
            
            completion = self.generate(prompt)
            response = self.validate(completion)
            
            ending = response[-20:].lower()
            if "invalid" in ending:
                self.prior.set_examples([
                    "Here is an example of an invalid summarization " +
                    f"that you should not generate:\n{completion}"])
            elif "valid" in ending:
                break
            
        return completion
    
    def generate(self, prompt: str) -> str:
        return self.prior.generate(prompt)
    
    def validate(self, prompt: str) -> str:
        return self.validator.generate(prompt)