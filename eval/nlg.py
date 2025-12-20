from eval.evaluator import Evaluator
from tqdm import tqdm
import json
from collections import Counter
import numpy as np
import torch
import sacrebleu
from typing import cast

from models.model import LLMTask

class NLG_Evaluator(Evaluator):
  def __init__(self, component: LLMTask, filepath: str, prompt: dict) -> None:
    """Initialize NLG Evaluator.
    Args:
      component (LLMTast): model for the nlg task.
      filepath (str): path of the dataset.
      prompt (dict): dict containing prompts for intent.  
    """
    super().__init__(component, filepath, prompt)
    self.pred_states, self.gt_states = self.get_pred_gt()

  def get_pred_gt(self) -> tuple:
    """Compute all pred_states and extract gt_states from test set.
    Returns:
      tuple: predicitons and ground truths.
    """
    pred_states = []
    gt_states = []

    for sample in tqdm(self.test_set, desc="Evaluating NLG"):
      intent = sample["intent"]
      self.component.change_system_prompt(self.prompt["prompt"]["main"] + self.prompt["prompt"][intent])

      pred = self.component.generate(json.dumps(sample["input"]))
      pred_states.append(pred)
      gt_states.append(sample["annotation"])

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

    return {
      "bleu": bleu_score,
      "f1": float(avg_f1),
    }
