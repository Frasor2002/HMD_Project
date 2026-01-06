from eval.evaluator import Evaluator
from tqdm import tqdm
import json
from collections import Counter
import numpy as np
import sacrebleu
from typing import Any
import re

from agent.preproc import Preproc
import os

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(EVAL_DIR, "results", "preproc_results.json")
STATE_PATH = os.path.join(EVAL_DIR, "temp", "preproc_state.json")


class PreprocEvaluator(Evaluator):
  def __init__(self, preproc: Preproc, filepath: str, prompt: dict) -> None:
    """Initialize the Preproc evaluator.
    Args:
      preproc (Preproc): model for preproc task.
      filepath (str): path of the dataset.
      prompt (dict): dict containing prompts for intent.
    """
    super().__init__(preproc, filepath, prompt)
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

    for sample in tqdm(remaining_samples, desc="Evaluating Preproc", initial=start_idx, total=len(self.test_set)):
      pred = self.component.generate(sample["utterance"], validate=False)
      try:
        pred = json.loads(pred)
      except Exception:
        # Gemma gives out output in a different format
        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, pred, re.DOTALL)
        if match:
          pred = json.loads(match.group(1))
        else: pred = []


      pred_states.append(pred)
      gt_states.append(sample["annotation"])

      # Save state every k samples
      if len(pred_states) % 1 == 0:
        self.save_eval_state(pred_states, gt_states, STATE_PATH)
    
    # Save last batch of samples
    self.save_eval_state(pred_states, gt_states, STATE_PATH)

    return pred_states, gt_states
  

  def normalize_to_string(self, x):
    """Flattens the output (which might be a list or JSON) into a single string."""
    if isinstance(x, str):
      # Attempt to parse if it looks like a JSON list representation
      try:
        x = json.loads(x)
      except Exception:
        pass

    if isinstance(x, list):
      # Recursively normalize elements and join
      return " ".join([str(e).strip().lower() for e in x])
    
    return str(x).strip().lower()
  
  def _compute_f1(self, pred_tokens: list, ref_tokens: list) -> float:
    """Compute F1 score based on token overlap.
    Args:
      pred_tokens (list): predicted tokens.
      ref_tokens (list): gt tokens.
    Returns:
      float, f1 score.
    """
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
      return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(ref_tokens)
    if precision + recall == 0:
      return 0.0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
  
  def _compute_sample_metrics(self, pred_str: str, ref_str: str) -> dict:
    """Computes F1 and Sentence BLEU for a single sample.
    Args:
      pred_str (str): flattened prediction string.
      ref_str(str): flatteneted gt string.
    Returns:
      dict: report of metrics.
    """
    pred_tokens = pred_str.split()
    ref_tokens = ref_str.split()

    if not pred_tokens or not ref_tokens:
      f1 = 0.0 if (pred_tokens or ref_tokens) else 1.0 # Both empty = match
    else:
      f1 = self._compute_f1(pred_tokens, ref_tokens)

    if not pred_str and not ref_str:
      bleu = 100.0
    else:
      bleu_res = sacrebleu.sentence_bleu(pred_str, [ref_str])
      bleu = bleu_res.score # This is 0-100 scale

    result = {
      "f1": f1,
      "bleu": bleu
    }

    return result
  
  def evaluate(self) -> dict:
    """Evaluate using F1 and BLEU score.
    Returns:
      dict: results report.
    """
        
    if not self.gt_states:
      return {"mean_score": 0.0, "f1": 0.0, "bleu": 0.0}

    sample_scores = []

    for pred_raw, expected_raw in zip(self.pred_states, self.gt_states):
      pred_str = self.normalize_to_string(pred_raw)
      ref_str = self.normalize_to_string(expected_raw)

      metrics = self._compute_sample_metrics(pred_str, ref_str)
      sample_scores.append(metrics)

    avg_f1 = np.mean([s["f1"] for s in sample_scores])
    avg_bleu = np.mean([s["bleu"] for s in sample_scores])

    metrics = {
      "f1": float(avg_f1),
      "bleu": float(avg_bleu)
    }

    self.save_results(metrics, RESULTS_PATH)
    return metrics
