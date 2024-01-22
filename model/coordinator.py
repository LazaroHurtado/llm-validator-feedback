from model.extractor_lm import ExtractorLM
from model.prior_lm import PriorLM
from model.validator_lm import ValidatorLM
from helpers.model_logger import ModelLogger

class Coordinator():
    LOGGER = ModelLogger("COORDINATOR")

    def __init__(self,
                 extractor: ExtractorLM,
                 prior: PriorLM,
                 validator: ValidatorLM,
                 retries: int = 3):
        self.extractor = extractor
        self.prior = prior
        self.validator = validator

        self.retries = retries

    @LOGGER.log_completion
    def call(self, prompt: str) -> str:
        extracted_constraints = self.extract(self.prior.meta_prompt)
        self.validator.set_constraints(extracted_constraints)

        for attempt in range(self.retries):
            self.LOGGER.info(f"Attempt {attempt+1} of {self.retries}")
            
            completion = self.generate(prompt)
            response = self.validate(completion)
            
            ending = response[-20:].lower()
            if "invalid" in ending:
                self.prior.set_examples([completion])
            elif "valid" in ending:
                break
            
        return completion
    
    def extract(self, prompt: str) -> str:
        return self.extractor.generate(prompt)
    
    def generate(self, prompt: str) -> str:
        return self.prior.generate(prompt)
    
    def validate(self, prompt: str) -> str:
        return self.validator.generate(prompt)