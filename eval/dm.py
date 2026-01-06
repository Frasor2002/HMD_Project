from typing import Dict
import json
from collections import defaultdict
from eval.evaluator import Evaluator
from agent.dm import DM
from tqdm import tqdm
import os

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(EVAL_DIR, "results", "dm_results.json")
STATE_PATH = os.path.join(EVAL_DIR, "temp", "dm_state.json")

class DM_Evaluator(Evaluator):
  def __init__(
    self, 
    dm: DM,
    filepath: str,
    prompt: Dict):
    """Initialize evaluator.
    Args:
      dm (DM): component to evaluate.
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
    # Resume state from file
    start_idx, pred_states, gt_states = self.resume_eval_state(STATE_PATH)
    # Check if eval is already done
    if start_idx >= len(self.test_set):
      return pred_states, gt_states

    remaining_samples = self.test_set[start_idx:]

    for sample in tqdm(remaining_samples, desc="Evaluating DM", initial=start_idx, total=len(self.test_set)):
      # Adapt prompt based on intent
      pred = self.component.generate(sample["ds"], validate = False)
      
      pred_states.append(pred)
      gt_states.append(sample["annotation"])

      # Save state every k samples
      if len(pred_states) % 10 == 0:
        self.save_eval_state(pred_states, gt_states, STATE_PATH)
    
    # Save last batch of samples
    self.save_eval_state(pred_states, gt_states, STATE_PATH)

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

    metrics = {
      "total_accuracy": total_accuracy,
      "class_accuracy": class_accuracy,
    }

    self.save_results(metrics, RESULTS_PATH)

    return metrics
  


