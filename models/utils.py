from typing import Dict, List, Optional, Any

from transformers import PreTrainedTokenizer
from huggingface_hub import login, whoami
import os

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


def gemma_prepare_text(
  prompt: str,
  tokenizer: PreTrainedTokenizer,
  messages: Optional[List[Dict[str, str]]] = None
) -> Any:
    """Prepare textual input for a gemma model.
  Args:
    prompt (str): textual prompt.
    tokenizer (PreTrainedTokenizer): tokenizer to tokenize the text into tokens.
    messages (Optional[List[Dict[str, str]]]): dialogue state.
  Returns:
    Any: prepared input for the gemma model.
  """
    if messages is None:
      messages = []
    conversation = messages + [{"role": "user", "content": prompt}]

    # Gemma does not have system role, needs to be merged with user
    if conversation and conversation[0]["role"] == "system":
      system_content = conversation.pop(0)["content"]
      for msg in conversation:
        if msg["role"] == "user":
          msg["content"] = f"{system_content}\n\n{msg['content']}"
          break
        

    return tokenizer.apply_chat_template(
      conversation,
      tokenize=False,
      add_generation_prompt=True,
    )


def login_to_hub() -> None:
  """Login to hugging face hub to load models."""
  # Check if already logged in
  try:
    user = whoami()
    print(f"hf user '{user['name']}' logged in.")
    return
  except Exception:
    pass
  
  # Login
  env_token = os.getenv("HF_TOKEN")
  if env_token:
    print("Logging in with HF_TOKEN...")
    login(token=env_token)
  else:
    print("WARNING: No authentication found. Set 'HF_TOKEN'.")
    