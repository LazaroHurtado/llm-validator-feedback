import logging

from model.prior_lm import PriorLM
from model.validator_lm import ValidatorLM
from model.coordinator import Coordinator
from commands import Args, ModelArgs, parse_args

LOG_FILE_PATH = './logger.out'

class Main():
    LOGGER = logging.getLogger("[MAIN]")

    @staticmethod
    def run(model: ModelArgs, prompt: str, args: Args = None):
        Main.LOGGER.debug(f"Running with\n\tmodel: {model.__dict__},\n\tprompt: {prompt}")
        
        prior = PriorLM(model)
        validator = ValidatorLM(model)

        coordinator = Coordinator(prior, validator)
        return coordinator.generate(prompt)

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(filename=LOG_FILE_PATH, level=args.log_level)

    output = Main.run(args.model, args.prompt, args)
    print(output)
