import textwrap
import torch

def format_meta_prompt(meta_prompt: str) -> str:
    return textwrap.dedent(meta_prompt).replace('\n',' ').strip()

def tokenize(prompt: str, tokenizer):
    return tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

def detokenize(tokens: str, tokenizer):
    return tokenizer.batch_decode(tokens)[0]

def generate(model, tokenizer, prompt: str, lm_config={}):
    tokenized_prompt = tokenize(prompt, tokenizer)
    inputs = tokenized_prompt | lm_config
    outputs = model.generate(**inputs)

    return detokenize(outputs, tokenizer)

def prepare_model_config(model_config: dict):
    dtype_str = model_config.get("torch_dtype", None)
    if dtype_str is not None:
        model_config["torch_dtype"] = getattr(torch, dtype_str)
    
    return model_config