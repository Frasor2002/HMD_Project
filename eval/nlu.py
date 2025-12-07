from typing import Any, Dict, Tuple, List
import json
from collections import defaultdict
from eval.evaluator import Evaluator
from models.model import LLMTask


class NLU_Evaluator(Evaluator):
  def __init__(
    self,
    nlu: LLMTask,
    filepath: str,
    prompt: dict
    ):
    """Initialize evaluator.
    Args:
      nlu (LLMTask): component to evaluate.
      filepath (str): test set filepath.
    """
    super().__init__(nlu, filepath, prompt)

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

    for sample in self.test_set:
      pred = self.component.generate(sample["utt"], sample["history"])
      pred_states.append(pred)
      gt_states.append(sample["annotation"])
    return pred_states, gt_states


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
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
      "f1_score": f1_score,
      "precision": precision,
      "recall": recall,
      "support": tp + fn
    }

    
  def evaluate(self) -> Dict[str, Any]:
    """
    Evaluate NLU predictions against ground-truth dialogue states.
    """
    if len(self.pred_states) != len(self.gt_states):
      raise ValueError("pred_states and gt_states must have the same length")

    if len(self.gt_states) == 0:
      return {
        "intent_accuracy": 0.0, 
        "intents_by_type": {}, 
        "slots_overall": {}, 
        "slots_by_type": {}
      }

    intent_correct = 0
    intent_type_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    slot_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    global_slot_stats = {'tp': 0, 'fp': 0, 'fn': 0}

    for pred, gt in zip(self.pred_states, self.gt_states):
      # Intent evaluation
      gt_intent = gt.get("intent", "UNKNOWN")
      pred_intent = pred.get("intent")            
      intent_type_stats[gt_intent]['total'] += 1
      if pred_intent == gt_intent:
        intent_correct += 1
        intent_type_stats[gt_intent]['correct'] += 1
      
      # Slot Evaluation
      gt_slots = gt.get("slots", {}) or {}
      pred_slots = pred.get("slots", {}) or {}

      # Union of slot names
      all_slots = set(pred_slots.keys()) | set(gt_slots.keys())

      for slot in all_slots:
        gt_val = gt_slots.get(slot)
        pred_val = pred_slots.get(slot)

        # False negative
        if slot in gt_slots and slot not in pred_slots:
          slot_stats[slot]['fn'] += 1
          global_slot_stats['fn'] += 1
        # False positive
        elif slot in pred_slots and slot not in gt_slots:
          slot_stats[slot]['fp'] += 1
          global_slot_stats['fp'] += 1
        else:
          if self._equal_slot(pred_val, gt_val):
            slot_stats[slot]['tp'] += 1
            global_slot_stats['tp'] += 1
          else:
            # wrong value predicted penalize both fp and fn
            slot_stats[slot]['fp'] += 1
            slot_stats[slot]['fn'] += 1
            global_slot_stats['fp'] += 1
            global_slot_stats['fn'] += 1

    
    intents_by_type = {}
    for intent, counts in intent_type_stats.items():
      total = counts['total']
      correct = counts['correct']
      intents_by_type[intent] = {
        "accuracy": correct / total if total > 0 else 0.0,
        "count": total,
        "correct": correct
      }

    slots_by_type = {}
    for slot_name, counts in slot_stats.items():
      slots_by_type[slot_name] = self._get_f1_precision_recall(
        counts['tp'], counts['fp'], counts['fn']
      )
    slots_overall = self._get_f1_precision_recall(
      global_slot_stats['tp'], global_slot_stats['fp'], global_slot_stats['fn']
    )
    
    return {
      "intent_accuracy": intent_correct / len(self.gt_states),
      "intents_by_type": intents_by_type,
      "slots_overall": slots_overall,
      "slots_by_type": slots_by_type
    }