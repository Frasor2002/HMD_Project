from typing import List, Tuple
import json
from models.model import LLMTask

class Evaluator:
  def __init__(
    self,
    component: LLMTask,
    filepath: str,
    prompt: dict
    ):
    """Initialize evaluator.
    Args:
      nlu (LLMTask): component to evaluate.
      filepath (str): test set filepath.
    """
    # Init compoenent to eval
    self.component = component
    self.prompt = prompt

    # Load test set
    self.test_set = self.load_test_set(filepath)

  def load_test_set(self, filepath: str) -> dict:
    """Load a new test set.
    Args:
      filepath (str): path of the test set.
    Returns:
      dict: loaded test set.
    """
    with open(filepath, "r") as f:
      test_set = json.load(f)
    return test_set


  
  

