from typing import List, Tuple, Any
import json
import os

class Evaluator:
  def __init__(
    self,
    component: Any,
    filepath: str,
    prompt: dict
    ):
    """Initialize evaluator.
    Args:
      nlu (Any): component to evaluate.
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
  
  def save_results(self, metrics: dict, filepath: str) -> None:
    """Save metrics to file,
    Args:
      metrics (dict): dict containing eval metrics.
      filepath (str): str containing file where to dump results
    """
    try:
      with open(filepath, 'w', encoding='utf-8') as f:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        json.dump(metrics, f, indent=2, ensure_ascii=False)
      print(f"Results saved cleanly to {filepath}")
    except Exception as e:
      print(f"Error saving results: {e}")
  
  def save_eval_state(self, preds: list, gts: list, filepath: str) -> None:
    """Save the current evaluation state to save time.
    Args:
      preds (list): new model predictions.
      gts (list): associated gts.
      filepath (str): filepath where to dump data.
    """
    data = [{"id": idx, "pred": p, "gt": g} for idx, (p,g) in enumerate(zip(preds, gts))]
    try:
      os.makedirs(os.path.dirname(filepath), exist_ok=True)
      with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
      print(f"Warning: Failed to save visualization file: {e}")
  
  def resume_eval_state(self, filepath: str) -> tuple:
    """Resume previous evaluation state.
    Args:
      filepath (str): filepath of old eval_state if present.
    Returns:
      tuple: contains starting sample idx, existing pred and existing gts
    """
    start_idx = 0
    preds = []
    gts = []
    if os.path.exists(filepath):
      try:
        print(f"Found existing file at {filepath}. Attempting to resume...")
        with open(filepath, 'r', encoding='utf-8') as f:
          data = json.load(f)   
          if isinstance(data, list) and len(data) > 0:
            for item in data:
              preds.append(item.get("pred"))
              gts.append(item.get("gt"))
            start_idx = len(preds)
            print(f"Successfully resumed. Skipping first {start_idx} samples.")
      except Exception as e:
        print(f"Warning: Could not load existing progress ({e}). Starting from scratch.")
        
    return start_idx, preds, gts


  
  

