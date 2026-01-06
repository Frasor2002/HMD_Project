from models.model import ModelLoader, LLMTask
import re
import json
from typing import Any, Optional, Union


def validate_sa(sa_out: str) -> str:
  """Validate sentiment analysis output.
  Args.
    sa_out (str): sentiment analysis output.
  Returns:
    str: validated output.
  """
  fallback_sentiment = "neutral"
  cleaned_out = sa_out.lower().strip()
  valid_labels = {"positive", "negative", "neutral"}
  if isinstance(cleaned_out, str) and cleaned_out in valid_labels:
    return cleaned_out

  print(f"wrong sa output {sa_out}")
  return fallback_sentiment

class SA:
  def __init__(self, loader: ModelLoader, prompt: dict) -> None:
    self.prompt = prompt
    self.llm = LLMTask(loader, prompt["prompt"])
  
  def generate(self, review: str, validate: bool = True) -> str:
    """Get label based on a single review.
    Args:
      review (str): single review.
      validate (bool): flag to set for validation.
    Returns:
      str: sentiment label.
    """
    raw_out = self.llm.generate(review)
    if validate: return validate_sa(raw_out)
    else: return raw_out

  def analyze(self, reviews: list) -> dict:
    """Analyze multiple reviews and return a report."""
    report = {"positive": 0, "negative": 0, "neutral": 0}
    for review in reviews:
      label = self.generate(review)
      report[label] += 1
    return report

