from typing import List, Dict, Tuple, Any

def _normalize_val(v: Any):
    """Normalize slot values for comparison."""
    if isinstance(v, str) and v.strip().lower() == "null":
        return None
    if v is None:
        return None
    # try to normalize numeric strings to int
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

def _equal_slot(a: Any, b: Any) -> bool:
    a_n = _normalize_val(a)
    b_n = _normalize_val(b)
    return a_n == b_n

def evaluate_nlu(pred_states: List[Dict], gt_states: List[Dict]) -> Dict:
    """
    Evaluate NLU predictions against ground-truth dialogue states.

    Args:
      pred_states: list of predicted dialogue states, each like {"intent": "...", "slots": {...}}
      gt_states:   list of ground-truth dialogue states, same format

    Returns:
      Dictionary with:
        - intent_accuracy: fraction of examples with correct intent
        - slot_accuracy: fraction of slot values correct (computed over all GT slots)
    """
    if len(pred_states) != len(gt_states):
        raise ValueError("pred_states and gt_states must have the same length")

    n = len(gt_states)
    if n == 0:
        return {"intent_accuracy": 0.0, "slot_accuracy": 0.0}

    intent_correct = 0
    total_slots = 0
    correct_slots = 0

    for pred, gt in zip(pred_states, gt_states):
        # intents
        if pred.get("intent") == gt.get("intent"):
            intent_correct += 1

        gt_slots = gt.get("slots", {}) or {}
        pred_slots = pred.get("slots", {}) or {}

        for slot_name, gt_val in gt_slots.items():
            total_slots += 1
            # predicted might miss the slot key -> treat as None
            pred_val = pred_slots.get(slot_name, None)
            if _equal_slot(pred_val, gt_val):
                correct_slots += 1

    return {
        "intent_accuracy": intent_correct / n,
        "slot_accuracy": (correct_slots / total_slots) if total_slots else 0.0
    }