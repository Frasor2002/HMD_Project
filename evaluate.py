import os
from models.model import LLMTask, ModelLoader
from models.registry import MODELS
from argparse import ArgumentParser, Namespace
import torch
import yaml
from eval.nlu import NLU_Evaluator
from eval.dm import DM_Evaluator
from typing import Any
from dotenv import load_dotenv
from models.utils import login_to_hub
# TODO create code to evaluate each component

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(PROJ_DIR, "eval", "test_set")

PROMPT_DIR = os.path.join(PROJ_DIR, "prompt")



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
    "-c", "--component",
    type=str,
    default="nlu",
    help="Component to evaluate.",
  )
  return parser.parse_args()

def get_evaluator(component) -> Any:
  name_to_class = {
    "nlu": NLU_Evaluator,
    "dm": DM_Evaluator
  }

  return name_to_class[component]


def eval() -> None:
  """Execute the agent."""
  load_dotenv()
  login_to_hub()

  args = parse_args()

  device = "auto"
  component_name = args.component

  # Get test set path
  test_set_path = os.path.join(TEST_DIR, component_name + ".json")
  prompt_path = os.path.join(PROMPT_DIR, component_name + ".yaml")
  
  with open(prompt_path, "r", encoding="utf-8") as file:
    prompt = yaml.safe_load(file)
  
  # Init component
  model_loader = ModelLoader(args.model, device)

  if component_name in ["dm", "nlg"]:
    component = LLMTask(model_loader, prompt["prompt"]["main"])
  else:
    component = LLMTask(model_loader, prompt["prompt"])

  # Get evaluator class based on which component to evaluate
  eval_class = get_evaluator(component_name)
  evaluator = eval_class(component, test_set_path, prompt)

  # Initialize and run evaluator
  results = evaluator.evaluate()
  
  print(results)

if __name__ == "__main__":
  eval()