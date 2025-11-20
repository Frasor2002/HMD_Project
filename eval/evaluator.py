from typing import List, Tuple
import json
from models.model import LLMTask

class Evaluator:
  def __init__(
      self,
      component: LLMTask,
      filepath: str,
      ):
    """Initialize evaluator.
    Args:
      nlu (LLMTask): component to evaluate.
      filepath (str): test set filepath.
    """
    # Init compoenent to eval
    self.component = component    

    # Load test set
    self.test_set = self.load_test_set(filepath)

    # Get predictions
    states = self.get_pred_gt()
    self.pred_states = states[0]
    self.gt_states = states[1]


  def load_test_set(self, filepath: str) -> dict:
    """Load a new test set."""
    with open(filepath, "r") as f:
      test_set = json.load(f)
    return test_set


  def get_pred_gt(self) -> Tuple[List]:
    """Compute all pred_states and extract gt_states from test set."""
    pred_states = []
    gt_states = []

    for sample in self.test_set:
      pred = self.component.generate(sample["utt"])
      pred_states.append(pred)
      gt_states.append(sample["annotation"])
    return pred_states, gt_states
  

