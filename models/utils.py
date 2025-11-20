from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer


def hf_prepare_text(
  prompt: str,
  tokenizer: PreTrainedTokenizer,
  messages: Optional[List[Dict[str, str]]] = None
):
  """Prepare textual input for a hugging face model.
  Args:
    prompt (str): textual prompt.
    tokenizer (PreTrainedTokenizer): tokenizer to tokenize the text into tokens.
    messages (Optional[List[Dict[str, str]]]): dialogue state.
    n_exchanges (int): number of previous exchanges to be used for context.
  """

  if messages is None:
    messages = []
  messages.append({"role": "user", "content": prompt})

  text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
  )

  return text