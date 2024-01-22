import logging
import random

from typing import Optional
from datasets import load_dataset
from model.extractor_lm import ExtractorLM
from model.prior_lm import PriorLM
from model.validator_lm import ValidatorLM
from model.coordinator import Coordinator
from commands import Args, ModelArgs, parse_args

class Main():
    LOGGER = logging.getLogger("MAIN")

    @staticmethod
    def run(model: ModelArgs,
            prompt: Optional[str],
            dataset: Optional[str],
            args: Args = None):
        if prompt is None and dataset is None:
            raise RuntimeError("Either prompt or dataset must be provided")
        elif prompt is None:
            dataset = load_dataset(dataset, split="train")
            rnd_index = random.randint(0, len(dataset) - 1)
            prompt = dataset[rnd_index]["report"]

        Main.LOGGER.debug(f"Running with model: {model.__dict__}")
        
        extractor = ExtractorLM(model)
        prior = PriorLM(model)
        validator = ValidatorLM(model)

        coordinator = Coordinator(extractor, prior, validator)
        return coordinator.call(prompt)

if __name__ == "__main__":
    args = parse_args()
    
    output = Main.run(args.model, args.prompt, args.dataset, args)