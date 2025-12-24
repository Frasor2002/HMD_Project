from models.model import ModelLoader, LLMTask
import re
import json
from typing import Any, Optional, Union

class NLG:
  """Natural Language Generator component."""
  def __init__(self, loader: ModelLoader, prompt: dict) -> None:
    """Initialize the component.
    Args:
      loader (ModelLoader): model loader for component.
      prompt (dict): prompt for llm
    """
    self.prompt = prompt
    self.llm = LLMTask(loader, prompt["prompt"]["main"])
  
  def set_prompt(self, intent_name: str, additional_tuning: Optional[str] = None) -> None:
    """Sets the prompt given the current intent.
    Args:
      intent_name (str): current intent name.
      additional_tuning (Optional[str]): additional prompt to further tune the response.
    """
    base_prompt = self.prompt["prompt"]["main"]
    intent_based_prompt = self.prompt["prompt"][intent_name]
    additional_prompt = "\n" + self.prompt["prompt"][additional_tuning] if additional_tuning else ""

    final_prompt = base_prompt + "\n" + intent_based_prompt + additional_prompt
    self.llm.change_system_prompt(final_prompt)

  def eval_generate(self, intent_name: str, formatted_input: str) -> str:
    """Generation for eval data that already has formatted input.
    Args:
      intent_name (str): intent_name to set the prompt.
      formatted_input (str): already formatted input.
    Returns:
      str: lexicalized response.
    """
    out = self.llm.generate(formatted_input)
    return out

  def generate(self, nba:str, ds: dict, ek: Optional[dict], mi: bool, additional_tuning: Optional[str] = None) -> str:
    """Get lexicalized output for the user.
    Args:
      nba (str): next best action.
      ds (dict): dialogue state.
      ek: (Optional[dict]): external knowledge.
      mi (bool): flag for multiple intents.
      additional_tuning (Optional[str]): additional prompt to further tune the response.
    Returns:
      str: generated response.
    """
    intent_name = ds.get("intent", "out_of_domain")
    self.set_prompt(intent_name, additional_tuning)
    ds_string = json.dumps(ds)
    if ek is not None: ek_string = json.dumps(ek)
    else: ek_string = ek
    nlg_input = f"NBA: {nba}\nDS: {ds_string}\nEK: {ek_string}\n MI: {mi}"
    print(nlg_input)

    out = self.llm.generate(nlg_input)
    return out

