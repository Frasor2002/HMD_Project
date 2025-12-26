from eval.evaluator import Evaluator
from tqdm import tqdm
import json
from collections import Counter, defaultdict
import numpy as np
from typing import Any

from agent.sa import SA
import os

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(EVAL_DIR, "results", "sa_results.json")
STATE_PATH = os.path.join(EVAL_DIR, "temp", "sa_state.json")

class SA_Evaluator(Evaluator):
  def __init__(self, sa: SA, filepath: str, prompt: dict):
    """Initialize the SA evaluator.
    Args:
      sa (SA): model for sentiment analysis task.
      filepath (str): path of the dataset.
      prompt (dict): dict containing prompts for intent.
    """
    super().__init__(sa, filepath, prompt)
    self.pred_states, self.gt_states = self.get_pred_gt()

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

    for sample in tqdm(remaining_samples, desc="Evaluating SA", initial=start_idx, total=len(self.test_set)):
      pred = self.component.generate(sample["review"], validate=False)
      pred_states.append(pred)
      gt_states.append(sample["annotation"])

      # Save state every k samples
      if len(pred_states) % 1 == 0:
        self.save_eval_state(pred_states, gt_states, STATE_PATH)
    
    # Save last batch of samples
    self.save_eval_state(pred_states, gt_states, STATE_PATH)

    return pred_states, gt_states
  
  @staticmethod
  def _sentiment_is_equal(pred: str, gt: str) -> bool:
    """Compare a prediction and a gt sentiment.
    Args:
      pred (str): predicted sentiment.
      gt (str): ground truth sentiment.
    Returns:
      bool: true if they are the same, false if not.
    """
    return str(pred).replace('"', '').replace("'", "").strip().lower() == str(gt).replace('"', '').replace("'", "").strip().lower()

  def evaluate(self) -> dict:
    """Evaluate using accuracy of analysis.
    Returns:
      dict, results report.
    """
    if not self.gt_states:
      return {"total_accuracy": 0.0, "class_accuracy": {}}

    total_correct = 0
    total_samples = len(self.gt_states)
    sentiment_counts = defaultdict(int)
    sentiment_hits = defaultdict(int)

    for pred, gt in zip(self.pred_states, self.gt_states):
      sentiment_counts[gt] += 1

      if self._sentiment_is_equal(pred, gt):
        total_correct += 1
        sentiment_hits[gt] += 1

    total_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    class_accuracy = {}
    for sentiment, count in sentiment_counts.items():
      hits = sentiment_hits[sentiment]
      acc = hits / count if count > 0 else 0.0
      class_accuracy[sentiment] = acc

    metrics = {
      "total_accuracy": total_accuracy,
      "class_accuracy": class_accuracy,
    }

    self.save_results(metrics, RESULTS_PATH)
    return metrics
