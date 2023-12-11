import logging

from model.base_lm import BaseLM

class Coordinator():
    LOGGER = logging.getLogger("[COORDINATOR]")

    def __init__(self, prior: BaseLM, validator: BaseLM, retries: int = 3):
        self.prior = prior
        self.validator = validator

        self.retires = retries

    # TODO: add validator feedback with retries
    def generate(self, prompt: str) -> str:
        return self.prior.prompt_completion(prompt)