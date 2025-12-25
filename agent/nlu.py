from models.model import ModelLoader, LLMTask
import json
import re
from typing import Any, Optional


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
    # Clean artifacts (Gemma)
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


class NLU:
  """Natural Language Understanding component to extract intent and slots."""
  def __init__(self, loader: ModelLoader, prompt: dict) -> None:
    """Initialize component.
    Args:
      loader (ModelLoader): model loader for component.
      prompt (dict): prompts for component.
    """
    self.loader = loader
    self.prompt = prompt
    self.llm = LLMTask(loader, prompt["prompt"])
  
  def generate(self, nlu_input: str, history: Optional[list] = None, validate: bool = True) -> Any:
    """Given an input output the intent and slots.
    Args:
      nlu_input (str): string where intent and slots must be extracted.
      history (Optional[list]): conversation history.
      validation (bool): flag to set to validate output.
    Returns:
      Any: if validation is set return a dict with intents and slots otherwise we return a string. 
    """
    if history is None: history = []

    raw_out = self.llm.generate(nlu_input, history)
    if validate: return validate_nlu(raw_out)
    return raw_out

