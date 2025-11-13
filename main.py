from dataset import GameDataset

import argparse
from argparse import ArgumentParser, Namespace
import torch

from models import MODELS, ModelLoader
from components.nlu.nlu import NLU

def parse_args() -> Namespace:
    """Parse and return command line args.
    Returns:
        Namespace: command line args.
    """

    parser = ArgumentParser(
        description="Execute dialogue agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
        default=2,
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
    response = model.extract_intent_slots(user_input)
    print(f"\nnlu: {response}")


if __name__ == "__main__":
  args = parse_args()

  data = GameDataset("dataset/steam_dataset.example.feather")
  #print(data.df.head())

  model_loader = ModelLoader(args.model, args.device)
  nlu = NLU(model_loader, n_exchanges=0) #No history to avoid allucination args.n_exchanges
  chat(nlu)