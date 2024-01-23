from logging import Logger, FileHandler, Formatter, getLevelName
from os import environ
from time import gmtime, monotonic, strftime

class ModelLogger(Logger):
    DEFAULT_LOG_FILE_PATH = './logger.out'
    PROMPT_LOG = "\n".join([
        "TOTAL TIME: {timeElapsed}",
        "PROMPT:\n{prompt}",
        "COMPLETION:\n{completion}",
        ])
    
    def __init__(self, logger_name: str, log_file: str = DEFAULT_LOG_FILE_PATH):
        super().__init__(logger_name, level=self.get_level())
        
        file_handler = FileHandler(log_file)
        formatter = Formatter(
            '[%(levelname)s;%(asctime)s][%(name)s] - %(message)s',
            datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)

        self.addHandler(file_handler)

    def get_level(self):
        log_level = environ.get("LOG_LEVEL", "INFO")
        level = getLevelName(log_level)

        if isinstance(level, str) and level.startswith("Level"):
            raise RuntimeError(f"Invalid log level: {log_level}")
        
        return level

    def log_completion(self, generate_fn):
        def wrapper(other_self, prompt: str) -> str:
            self.info(f"Starting generation")

            start_time = monotonic()
            completion = generate_fn(other_self, prompt)
            end_time = monotonic()

            elapsed_time = end_time - start_time
            formatted_time = strftime("%H:%M:%S", gmtime(elapsed_time))

            log = self.PROMPT_LOG.format(
                prompt=prompt,
                completion=completion,
                timeElapsed=formatted_time)
            self.info(log)

            return completion
        return wrapper