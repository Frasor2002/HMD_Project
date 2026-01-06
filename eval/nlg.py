from eval.evaluator import Evaluator
from tqdm import tqdm
import json
from collections import Counter
import numpy as np
import sacrebleu

from agent.nlg import NLG
import os

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(EVAL_DIR, "results", "nlg_results.json")
STATE_PATH = os.path.join(EVAL_DIR, "temp", "nlg_state.json")

class NLG_Evaluator(Evaluator):
  def __init__(self, nlg: NLG, filepath: str, prompt: dict) -> None:
    """Initialize NLG Evaluator.
    Args:
      nlg (LLMTast): model for the nlg task.
      filepath (str): path of the dataset.
      prompt (dict): dict containing prompts for intent.  
    """
    super().__init__(nlg, filepath, prompt)
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

    for sample in tqdm(remaining_samples, desc="Evaluating NLG", initial=start_idx, total=len(self.test_set)):
      intent = sample["intent"]
      pred = self.component.eval_generate(intent, json.dumps(sample["input"]))
      pred_states.append(pred)
      gt_states.append(sample["annotation"])

      # Save state every k samples
      if len(pred_states) % 10 == 0:
        self.save_eval_state(pred_states, gt_states, STATE_PATH)
    
    # Save last batch of samples
    self.save_eval_state(pred_states, gt_states, STATE_PATH)

    return pred_states, gt_states


  def _compute_f1(self, pred: str, refs: list) -> float:
    """Compute best f1 score among references for a sample.
    Args:
      pred (str): prediction to test.
      refs (list): list of gts.
    Returns:
      float: f1 score.
    """
    def get_f1(p_tokens: list, r_tokens: list) -> float:
      """Simple implementation to compute f1 score.
      Args:
        p_tokens (list): list of prediction tokens.
        r_tokens (list): list of reference tokens.
      Returns:
        float: f1 score.
      """
      common = Counter(p_tokens) & Counter(r_tokens)
      num_same = sum(common.values())
      if num_same == 0:
        return 0.0
      
      precision = 1.0 * num_same / len(p_tokens)
      recall = 1.0 * num_same / len(r_tokens)
      f1 = (2 * precision * recall) / (precision + recall)
      return f1

    # Tokenize the prediction
    pred_toks = pred.strip().lower().split()
    # Compute ref scores and keep best one
    scores = []
    for ref in refs:
      ref_toks = ref.strip().lower().split()
      scores.append(get_f1(pred_toks, ref_toks))
            
    return max(scores) if scores else 0.0
  
  def evaluate(self) -> dict:
    """Evaluate the nlg using reference strings."""

    if not self.gt_states:
      return {"bleu": 0.0, "f1": 0.0}
    
    # Computing f1 score
    f1_scores = [
      self._compute_f1(pred, refs) 
      for pred, refs in zip(self.pred_states, self.gt_states)
    ]
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0

    # Computing BLEU score
    bleu_score = 0.0
    # sacrebleu expects list of reference differently
    max_refs = max(len(refs) for refs in self.gt_states)
    transposed_refs = []
    for i in range(max_refs):
      ref_list = []
      for refs in self.gt_states:
        ref_list.append(refs[i] if i < len(refs) else "")
      transposed_refs.append(ref_list)

    bleu = sacrebleu.corpus_bleu(self.pred_states, transposed_refs)
    bleu_score = bleu.score

    metrics = {
      "bleu": bleu_score,
      "f1": float(avg_f1),
    }

    self.save_results(metrics, RESULTS_PATH)


    return metrics
