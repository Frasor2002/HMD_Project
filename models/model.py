import torch
from typing import List, Dict
from transformers import AutoTokenizer

from .registry import MODELS


class ModelLoader:
  """Class wrapper around chosen llm model and tokenizer."""

  def __init__(self, model_name: str, device: str = "cpu") -> None:
    """Load a model and its tokenizer.
    Args:
      model_name (str): name of the model to load.
      device (str): device where to load the model.
    """

    if model_name not in MODELS:
      raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}.")
    
    model_id, init_model, prepare_text = MODELS[model_name]

    print(f"Loading tokenizer and model: {model_name}.")
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.model = init_model(model_id, dtype="auto", device_map=device)

    self.model_id = model_id
    self.model_name = model_name
    self.device = device
    self.prepare_text_fun = prepare_text



class LLMTask:
  """Class to load and interact with the chosen LLM model for a specific task."""

  def __init__(
      self, 
      model_loader: ModelLoader,
      task_prompt: str,
      n_exchanges: int = 2
    ) -> None:
    """Initialize LLM to be used."""
    
    self.model = model_loader.model
    self.tokenizer = model_loader.tokenizer
    self.prepare_text_fun = model_loader.prepare_text_fun
    self.model_id = model_loader.model_id
    self.model_name = model_loader.model_name
    self.device = model_loader.device

    self.n_exchanges = n_exchanges

    # Dialogue state initialization
    self.messages: List[Dict[str, str]] = [
      {"role": "system", "content": task_prompt}
    ]

  
  def prepare_text(self, prompt: str) -> str:
    """Prepare prompt for the model depending on the given model.
    Args:
      prompt (str): prompt to prepare.
    Returns:
      str: model inputs.
    """
    return self.prepare_text_fun(prompt,
                                self.tokenizer,
                                self.messages,
                                self.n_exchanges)
  

  def generate(self, prompt: str, max_new_tokens: int = 10000) -> str:
    """Generate output given user prompt.
    Args:
      prompt (str): user prompt after which the model generates.
      max_new_tokens (str): maximum number of tokens to use to generate.
    Returns:
      str: generated response.
    """
    text = self.prepare_text(prompt)
    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

    with torch.no_grad():
      generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens).cpu()

    # Decode ids
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

    # Update message history
    self.messages.append({"role": "user", "content": prompt})
    self.messages.append({"role": "assistant", "content": content})

    # Clear message history after n_exchanges (keep system directive)
    max_messages = self.n_exchanges * 2 + 1
    if len(self.messages) > max_messages:
      self.messages = [self.messages[0]] + self.messages[-self.n_exchanges * 2 :]

    return content



