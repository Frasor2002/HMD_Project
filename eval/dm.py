from typing import Any, Dict, Tuple, List
import json
from collections import defaultdict
from eval.evaluator import Evaluator
from models.model import LLMTask
from tqdm import tqdm

class DM_Evaluator(Evaluator):
  def __init__(
    self, 
    dm: LLMTask,
    filepath: str,
    prompt: Dict):
    """Initialize evaluator.
    Args:
      nlu (LLMTask): component to evaluate.
      filepath (str): test set filepath.
      prompt (dict): prompt for the task.
    """
    super().__init__(dm, filepath, prompt)

    # Get predictions
    pred_states, gt_states = self.get_pred_gt()
    self.pred_states = pred_states
    self.gt_states = gt_states

  
  def get_pred_gt(self) -> tuple:
    """Compute all pred_states and extract gt_states from test set.
    Returns:
      tuple: predicitons and ground truths.
    """
    pred_states = []
    gt_states = []

    for sample in tqdm(self.test_set, desc="Evaluating DM"):
      # Adapt prompt based on intent
      intent = sample["ds"]["intent"]
      self.component.change_system_prompt(self.prompt["prompt"]["main"] + self.prompt["prompt"][intent])

      pred = self.component.generate(sample["ds"])
      print(sample['ds'])
      print(pred)
      pred_states.append(pred)
      gt_states.append(sample["nba"])
    return pred_states, gt_states
  
  @staticmethod
  def _action_is_equal(pred: str, gt: str) -> bool:
    """Compare a prediction and a gt action.
    Args:
      pred (str): predicted action.
      gt (str): ground truth action.
    Returns:
      bool: true if they are the same, false if not.
    """
    return pred.strip().lower() == gt.strip().lower()



  def evaluate(self) -> dict:
    """Compute dm accuracy and return total and finegrained accuracy for every action."""
    if not self.gt_states:
      return {"total_accuracy": 0.0, "class_accuracy": {}}

    total_correct = 0
    total_samples = len(self.gt_states)
    action_counts = defaultdict(int)
    action_hits = defaultdict(int)

    for pred, gt in zip(self.pred_states, self.gt_states):   
      action_counts[gt] += 1

      if self._action_is_equal(pred, gt):
        total_correct += 1
        action_hits[gt] += 1

    total_accuracy = total_correct / total_samples

    class_accuracy = {}
    for action, count in action_counts.items():
      hits = action_hits[action]
      acc = hits / count if count > 0 else 0.0
      class_accuracy[action] = acc

    return {
      "total_accuracy": total_accuracy,
      "class_accuracy": class_accuracy,
    }
  


