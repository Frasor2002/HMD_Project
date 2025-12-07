from typing import Dict, List, Optional, Any

from transformers import PreTrainedTokenizer


def hf_prepare_text(
  prompt: str,
  tokenizer: PreTrainedTokenizer,
  messages: Optional[List[Dict[str, str]]] = None
) -> Any:
  """Prepare textual input for a hugging face model.
  Args:
    prompt (str): textual prompt.
    tokenizer (PreTrainedTokenizer): tokenizer to tokenize the text into tokens.
    messages (Optional[List[Dict[str, str]]]): dialogue state.
  Returns:
    Any: prepared input for the hugging face model.
  """

  if messages is None:
    messages = []
  conversation = messages + [{"role": "user", "content": prompt}]

  text = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True,
  )

  return text