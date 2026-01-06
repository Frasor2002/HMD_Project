from typing import Any, Callable, Dict, Tuple
from functools import partial
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from .utils import hf_prepare_text, gemma_prepare_text

bnb_4bit = BitsAndBytesConfig(
    load_in_4bit=True
)

# To access llama and gemma, must accept terms of service at the following pages:
# https://huggingface.co/google/gemma-2-9b-it
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct


# The tuple contains the model name, a partial function with the model specific 
# arguments, the method to prepare the input text
MODELS: Dict[str, Tuple[str, Callable[..., Any], Callable[..., Any]]] = {
    "qwen3": (
      "Qwen/Qwen3-4B-Instruct-2507",
      partial(AutoModelForCausalLM.from_pretrained, trust_remote_code=True, quantization_config=bnb_4bit),
      hf_prepare_text
    ),
    "mistral": (
      "mistralai/Mistral-7B-Instruct-v0.3",
      partial(AutoModelForCausalLM.from_pretrained, trust_remote_code=True, quantization_config=bnb_4bit),
      hf_prepare_text
    ),
    # Gated models
    "llama3": (
      "meta-llama/Meta-Llama-3.1-8B-Instruct",
      partial(AutoModelForCausalLM.from_pretrained, trust_remote_code=True, quantization_config=bnb_4bit),
      hf_prepare_text
    ),
    "gemma": (
      "google/gemma-2-9b-it",
      partial(AutoModelForCausalLM.from_pretrained, trust_remote_code=True, quantization_config=bnb_4bit),
      gemma_prepare_text
    )
}