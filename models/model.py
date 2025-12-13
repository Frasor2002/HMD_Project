import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer
from models.registry import MODELS


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
      system_prompt: str
    ) -> None:
    """Initialize LLM to be used.
    Args:
      model_loader (ModelLoader): model loader that handles loading model and tokenizer.
      system_prompt (str): describes in detail the task that the llm must do.
    """
    
    self.model = model_loader.model
    self.tokenizer = model_loader.tokenizer
    self.prepare_text_fun = model_loader.prepare_text_fun
    self.model_id = model_loader.model_id
    self.model_name = model_loader.model_name
    self.device = model_loader.device
    self.system_prompt = system_prompt

  def change_system_prompt(self, new_prompt: str) -> None:
    """Change the system prompt to dynamically adjust the task based on intent.
    Args:
      new_prompt (str): new system prompt to load.
    """
    self.system_prompt = new_prompt



  def prepare_text(self, prompt: str) -> Any:
    """Prepare prompt for the model depending on the given model.
    Args:
      prompt (str): prompt to prepare.
    Returns:
      Any: model inputs.
    """
    return self.prepare_text_fun(prompt,
                                self.tokenizer,
                                self.messages)
  

  def generate(self, prompt: str, history: list = [], max_new_tokens: int = 512) -> str:
    """Generate output given user prompt.
    Args:
      prompt (str): user prompt after which the model generates.
      max_new_tokens (str): maximum number of tokens to use to generate.
    Returns:
      str: generated response.
    """
    # Reset conversation
    self.messages: List[Dict[str, str]] = [
      {"role": "system", "content": self.system_prompt}
    ]
    # Add history if passed
    self.messages += history

    text = self.prepare_text(prompt)
    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

    with torch.no_grad():
      generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens).cpu()

    # Decode ids
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

    return content



