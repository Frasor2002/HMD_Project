from typing import List, Dict, Tuple, Any
import json
from components.nlu.nlu import NLU
from collections import defaultdict

class NLU_Evaluator:
  def __init__(
      self,
      nlu: NLU,
      filepath: str = "test_set/test.json",
      ):
    """Initialize evaluator.
    Args:
      nlu (LLMTask): component to evaluate.
      filepath (str): test set filepath.
    """
    self.nlu = nlu

    # Load test set
    self.load_test_set(filepath)

    # Get predictions



  def load_test_set(self, filepath: str):
    """Load a new test set."""
    with open(filepath, "r") as f:
      self.test_set = json.load(f)


  def get_pred_gt(self):
    """Compute all pred_states and extract gt_states from test set."""
    self.pred_states = []
    self.gt_states = []

    for sample in self.test_set:
      pred = self.nlu.extract_intent_slots(sample["utt"])
      self.pred_states.append(pred)
      self.gt_states.append(sample["annotation"])


  @staticmethod
  def _normalize_val(v: Any):
    """Normalize slot values for comparison."""
    if isinstance(v, str) and v.strip().lower() == "null":
      return None
    if v is None:
      return None
    # Try to normalize numeric strings to int
    if isinstance(v, str):
      s = v.strip()
      if s.isdigit():
        return int(s)
      try:
        f = float(s)
        return f
      except Exception:
        return s.lower()
    if isinstance(v, (int, float)):
      return v
    return v

  @staticmethod
  def _equal_slot(a: Any, b: Any) -> bool:
    """Compare slots to see if they are equal."""
    a_n = NLU_Evaluator._normalize_val(a)
    b_n = NLU_Evaluator._normalize_val(b)
    return a_n == b_n
  
  @staticmethod
  def _get_f1_precision_recall(tp: int, fp: int, fn: int):
    """Compute metrics from counts."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fp > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
      "f1_score": f1_score,
      "precision": precision,
      "recall": recall,
      "total_instances": tp + fn
    }

    
  def evaluate(self) -> Dict:
    """
    Evaluate NLU predictions against ground-truth dialogue states.
    Args:
      pred_states: list of predicted dialogue states, each like {"intent": "...", "slots": {...}}.
      gt_states:   list of ground-truth dialogue states, same format.
    Returns:
      dict:
        - intent_accuracy: fraction of examples with correct intent.
        - slot_accuracy: fraction of slot values correct (computed over all GT slots).
    """
    if len(self.pred_states) != len(self.gt_states):
      raise ValueError("pred_states and gt_states must have the same length")

    if len(self.gt_states) == 0:
      return {"intent_accuracy": 0.0, "slot_accuracy": 0.0}

    # Store counts
    intent_counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    slot_counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    # Global counters
    intent_total = len(self.gt_states)
    intent_correct = 0
    slot_tp = 0
    slot_fp = 0
    slot_fn = 0

    # Iterate samples
    for pred, gt in zip(self.pred_states, self.gt_states):

      # Intent evaluation
      gt_intent = gt.get("intent")
      pred_intent = pred.get("intent")
      if pred_intent == gt_intent:
        intent_correct += 1
        # Keep track of specific intent performance
        intent_counts[gt_intent]['tp'] += 1
      else: # Wrong intent prediction
        # False negative for the true intent
        intent_counts[gt_intent]['fn'] += 1
        # False positive for predicted intent
        intent_counts[pred_intent]['fp'] += 1

      
      # Slot evaluation
      gt_slots = gt.get("slots", {}) or {}
      pred_slots = pred.get("slots", {}) or {}

      # All the slots present in this sample
      all_slots = set(pred_slots.keys()) | set(gt_slots.keys())

      # Iterate on all slots
      for slot in all_slots:
        #TODO slot eval
        pass

      for slot_name, gt_val in gt_slots.items():
        total_slots += 1
        # predicted might miss the slot key -> treat as None
        pred_val = pred_slots.get(slot_name, None)
        if self._equal_slot(pred_val, gt_val):
          correct_slots += 1

    return {
      "intent_accuracy": intent_correct / n,
      "slot_accuracy": (correct_slots / total_slots) if total_slots else 0.0
    }