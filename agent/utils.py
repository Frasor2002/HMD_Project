import json
import re

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
    

# Output validation for LLMs
def validate_preproc(preproc_out: str, user_input: str) -> list:
  """Validate the output of the preprocessor.
  Args:
    preproc_out (str): output of the preprocessor.
    user_input (str): user input for fallback.
  Return:
    list: validated output.
  """
  try:
    cleaned_out = preproc_out.strip()
    # Clean formatting artifacts
    if "```" in cleaned_out:
      match = re.search(r"```(?:json)?(.*?)```", cleaned_out, re.DOTALL)
      if match:
        cleaned_out = match.group(1).strip()
    
    # Parse json
    parsed = json.loads(cleaned_out)
    # Check that it is a list of strings as expected
    if not isinstance(parsed, list) or not all(isinstance(p, str) for p in parsed):
      print("Wrong preproc output type")
      return [user_input]
    # If user input is not an empty string, the list should not be empty
    if len(parsed) == 0 and len(user_input.strip()) > 0:
      print("Preproc returned empty output from user input")
      return [user_input]

    # Return parsed
    return parsed
  except Exception:
    print("Preproc wrong output format")
    return [user_input]

def validate_nlu(nlu_out: str) -> dict:
  """Validate the nlu output.
  Args:
    nlu_out (str): nlu output to be validated to get dialogue state.
  Returns:
    dict: validated nlu output.
  """
  fallback_out = {"intent": "out_of_domain", "slots": {}}
  try:
    cleaned_out = nlu_out.strip()
    # Clean artifacts
    if "```" in cleaned_out:
      match = re.search(r"```(?:json)?(.*?)```", cleaned_out, re.DOTALL)
      if match:
        cleaned_out = match.group(1).strip()
    # Try parsing
    parsed = json.loads(cleaned_out)
    # Check that the output schema is correct
    if isinstance(parsed, dict) and "intent" in parsed:
      # Make sure slots exist
      if "slots" not in parsed:
        parsed["slots"] = {}
      return parsed
    else:
      print("nlu output format incorrect")
      return fallback_out
  except Exception:
    print("nlu wrong output")
    return fallback_out

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
  
def validate_sa(sa_out: str) -> str:
  """Validate sentiment analysis output.
  Args.
    sa_out (str): sentiment analysis output.
  Returns:
    str: validated output.
  """
  fallback_sentiment = "neutral"
  cleaned_out = sa_out.lower().strip()
  valid_labels = {"positive", "negative", "neutral"}
  if isinstance(cleaned_out, str) and cleaned_out in valid_labels:
    return cleaned_out

  print("wrong sa output")
  return fallback_sentiment
