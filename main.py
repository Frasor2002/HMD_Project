from argparse import ArgumentParser, Namespace
import torch

from models.model import MODELS
from agent.agent import DialogueAgent


def parse_args() -> Namespace:
  """Parse and return command line args.
  Returns:
    Namespace: command line args.
  """
  parser = ArgumentParser()

  parser.add_argument(
    "-m", "--model",
    type=str,
    choices=MODELS.keys(),
    default="qwen3",
    help="Name of the llm model to use for each component.",
  )
  parser.add_argument(
    "-d", "--device",
    type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
    help="Device to run the model on.",
  )
  parser.add_argument(
    "-n", "--n_exchanges",
    type=int,
    default=3,
    help="Number of exchanges to keep in the conversation history.",
  )
  return parser.parse_args()


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
  args = parse_args()

  model = {
    "default": "qwen3",
    "preproc": "qwen3",
    "nlu": "qwen3",
    "dm": "qwen3",
    "sa": "qwen3"
  }

  dialogue_agent = DialogueAgent(model, args.device, args.n_exchanges)

  chat(dialogue_agent)

if __name__ == "__main__":
  main()