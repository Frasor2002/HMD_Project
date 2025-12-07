import os
from models.model import LLMTask, ModelLoader
from models.registry import MODELS
from argparse import ArgumentParser, Namespace
import torch
import yaml
from eval.nlu import NLU_Evaluator
from typing import Any
# TODO create code to evaluate each component

# /eval
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER = "test_set"
TEST_DIR = os.path.join(EVAL_DIR, FOLDER)

PROJ_DIR = os.path.dirname(EVAL_DIR)
PROMPT_DIR = os.path.join(EVAL_DIR, "prompt")



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
    help="Name of the llm model to use.",
  )
  parser.add_argument(
    "-d", "--device",
    type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
    help="Device to run the model on.",
  )
  parser.add_argument(
    "-c", "--component",
    type=str,
    default="nlu",
    help="Component to evaluate.",
  )
  return parser.parse_args()

def get_evaluator(component) -> Any:
  name_to_class = {
    "nlu": NLU_Evaluator
  }

  return name_to_class[component]


def eval() -> None:
  """Execute the agent."""
  args = parse_args()

  # Get test set path
  test_set_path = os.path.join(TEST_DIR, args.component + ".json")
  prompt_path = os.path.join(TEST_DIR, args.component + ".yaml")
  base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  
  with open(prompt_path, "r") as file:
    prompt = yaml.safe_load(file)
  
  # Init component
  model_loader = ModelLoader(args.model, args.device)
  component = LLMTask(model_loader, prompt["prompt"])

  # Get evaluator class based on which component to evaluate
  eval_class = get_evaluator(args.component)
  evaluator = eval_class(component, test_set_path, prompt)

  # Initialize and run evaluator
  results = evaluator.evaluate()
  
  print(results)

if __name__ == "__main__":
  eval()