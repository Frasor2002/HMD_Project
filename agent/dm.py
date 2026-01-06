from models.model import ModelLoader, LLMTask
import re
import json
from typing import Any, Optional, Union

def get_action(intent: str, slots: dict) -> str:
  """Given a ds return the action annotation.
  Args:
    intent (str): ds intent.
    slots (dict): ds slots.
  Returns:
    str: nba.
  """
  filled_slots = {k: v for k, v in slots.items() if v is not None}

  match intent:
    case "get_game_info":
      if slots.get("title") is None:
        action = "ask_for(title)"
      elif slots.get("info") is None:
        action = "ask_for(info)"
      else:
        info_val = slots.get("info")
        action = f"give_info(title, {info_val})"
  
    case "discover_game":
      if len(filled_slots) < 1:
        action = "ask_for(genre)"
      else:
        action = f"propose_game({', '.join(filled_slots.keys())})"
    
    case "compare_games":
      if slots.get("title1") is None:
        action = "ask_for(title1)"
      elif slots.get("title2") is None:
        action = "ask_for(title2)"
      elif slots.get("criteria") is None:
        action = "ask_for(criteria)"
      else:
        action = f"give_comparison({', '.join(filled_slots.keys())})"
    
    case "get_friend_games":
      if slots.get("name") is None:
        action = "ask_for(name)"
      else:
        action = "give_friend_games(name)"
    
    case "get_term_explained":
      if slots.get("term") is None:
        action = "ask_for(term)"
      else:
        action = "explain_term(term)"
    
    case "add_to_wishlist":
      if slots.get("title") is None:
        action = "ask_for(title)"
      else:
        action = "add_game(title)"
    
    case "remove_from_wishlist":
      if slots.get("title") is None:
        action = "ask_for(title)"
      else:
        action = "remove_game(title)"
    
    case "get_wishlist":
      action = "give_wishlist()"
    case "out_of_domain":
      action = "fallback()"
    case _:
      action = "fallback()"

  return action

class RuleBasedDM:
    """Wrapper for the rule-based logic to make it compatible with evaluator and agent."""
    
    def change_system_prompt(self, prompt: str):
      """Empty method for compatibility since we are not using an llm."""
      pass

    def generate(self, input_str: str) -> str:
      """Input is a string so conversion must be done for processing.
      Args:
        input_str (str): input ds in string format.
      Returns
        str: nba.
      """
      ds = json.loads(input_str)
      return get_action(**ds)
    
def validate_dm(dm_out: str) -> str:
  """Validate dm output.
  Args:
    dm_out (str): dm output.
  Returns:
    str: validated output.
  """
  fallback_nba = "fallback()"
  # Regex for how we expect action_name(input_slots)
  pattern = r'^([a-zA-Z_]\w*)\s*\('
  cleaned_out = dm_out.strip()
  match = re.match(pattern, cleaned_out)
  if match:
    return cleaned_out
  else:
    print("wrong dm output")
    return fallback_nba


class DM:
  """Dialogue Manager component that parsed dialogue state and returns next best action."""
  def __init__(self, loader: Union[ModelLoader, RuleBasedDM], prompt: dict) -> None:
    """Initialize component.
    Args:
      loader (Union[ModelLoader, RuleBasedDM]): model loader for component.
      prompt (dict): prompt for component, (not necessary with rulebased).
    """
    self.prompt = prompt

    if isinstance(loader, ModelLoader): self.llm = LLMTask(loader, prompt["prompt"]["main"])
    else: self.llm = loader

  def set_prompt(self, intent_name: str) -> None:
    """Sets the prompt given the current intent.
    Args:
      intent_name (str): current intent name.
    """
    base_prompt = self.prompt["prompt"]["main"]
    intent_based_prompt = self.prompt["prompt"][intent_name]
    final_prompt = base_prompt + "\n" + intent_based_prompt
    self.llm.change_system_prompt(final_prompt)

  def generate(self, ds: dict, validate: bool = True) -> str:
    """Get nba from ds.
    Args:
      ds (dict): dialogue state.
      validate (bool): flag to set to validate output or not.
    Returns:
      str: output from the llm.
    """
    intent_name = ds.get("intent", "out_of_domain")
    # Load intent-based prompt
    self.set_prompt(intent_name)
    ds_string = json.dumps(ds)
    raw_out = self.llm.generate(ds_string)
    if validate: return validate_dm(raw_out)
    return raw_out

