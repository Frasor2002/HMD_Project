import json

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