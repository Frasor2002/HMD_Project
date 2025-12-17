import torch
from dotenv import load_dotenv
from models.utils import login_to_hub
from agent.agent import DialogueAgent





def chat(model):
  """Interactive CLI session."""
  while True:
    user_input = input(f"\nUser: ")
    if user_input.lower() in {"exit", "quit"}:
      print("Closing...")
      break
    response = model.chat(user_input)
    print(f"Assistant: {response}")


def main() -> None:
  """Execute the agent."""

  # Load hf 
  load_dotenv()
  login_to_hub()

  device = "auto"
  n_exchanges = 2


  model = {
    "default": "qwen3",
    "preproc": "qwen3",
    "nlu": "qwen3",
    "dm": "qwen3",
    "sa": "qwen3"
  }

  dialogue_agent = DialogueAgent(model, device, n_exchanges)

  chat(dialogue_agent)

if __name__ == "__main__":
  main()