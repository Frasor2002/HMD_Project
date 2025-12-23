from models.model import ModelLoader, LLMTask
import json
import re
from typing import Any

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

class Preproc:
  """Class for the preprocessor component that splits text based on intent."""
  def __init__(self, loader: ModelLoader, prompt: dict) -> None:
    """Initialize component.
    Args:
      loader (ModelLoader): model loader for the component.
      prompt (dict): dictionary containing prompt for the component.
    """
    self.loader = loader
    self.prompt = prompt
    self.llm = LLMTask(loader, prompt["prompt"])
  
  def generate(self, user_input: str, validate: bool = True) -> Any:
    """Pass through the model to get splitted input.
    Args:
      user_input (str): user input string.
      validate (bool): flag to decide if output is validated or not.
    Returns:
      Any: the expected output should be a list of strings. Disabling validation could lead to some format errors from LLM.
    """
    raw_out = self.llm.generate(user_input)
    if validate: return validate_preproc(raw_out, user_input)
    return raw_out
    

