from commands import Args, ModelArgs, parse_args
from helpers.utils import model_from_args
from lm.coordinator import Coordinator
from lm.extractor_lm import ExtractorLM
from lm.prior_lm import PriorLM
from lm.validator_lm import ValidatorLM

from typing import Optional
from datasets import load_dataset

import logging
import random

class Main():
    LOGGER = logging.getLogger("MAIN")

    @staticmethod
    def run(model_args: ModelArgs,
            prompt: Optional[str],
            dataset: Optional[str],
            args: Args = None):
        if prompt is None and dataset is None:
            raise RuntimeError("Either prompt or dataset must be provided")
        elif prompt is None:
            dataset = load_dataset(dataset, split="test", trust_remote_code=True)
            rnd_index = random.randint(0, len(dataset) - 1)
            prompt = dataset[rnd_index]["report"]

        Main.LOGGER.debug(f"Running with model: {model_args.__dict__}")
        
        model = model_from_args(model_args)

        extractor = ExtractorLM(model)
        prior = PriorLM(model)
        validator = ValidatorLM(model)

        coordinator = Coordinator(extractor, prior, validator)
        return coordinator.call(prompt)

if __name__ == "__main__":
    args = parse_args()
    
    output = Main.run(args.model, args.prompt, args.dataset, args)