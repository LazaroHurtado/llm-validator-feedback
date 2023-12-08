from ctransformers import AutoModelForCausalLM
from typing import Optional

from commands import parse_args, ModelArgs

def main(model: Optional[ModelArgs],
         prompt: str = "Hello there!",
         **kwargs):
    if (model is None):
        exit(1)

    llm = AutoModelForCausalLM.from_pretrained(
        model["name"],
        model_file=model["file"],
        model_type=model["type"]
    )
    
    output = llm(prompt)
    print(output)

if __name__ == "__main__":
    args = parse_args()
    main(**args)