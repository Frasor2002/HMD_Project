import os
from models.model import LLMTask, ModelLoader
from models.registry import MODELS
from argparse import ArgumentParser, Namespace
import torch
import yaml
from eval.nlu import NLU_Evaluator
from eval.dm import DM_Evaluator
from eval.nlg import NLG_Evaluator
from eval.preproc import PreprocEvaluator
from eval.sa import SA_Evaluator
from typing import Any
from dotenv import load_dotenv
from models.utils import login_to_hub
from agent.preproc import Preproc
from agent.nlu import NLU
from agent.dm import DM, RuleBasedDM
from agent.nlg import NLG
from agent.sa import SA
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
    choices=list(MODELS.keys()) + ["rule_based"],
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

def get_evaluator(component: str) -> Any:
  name_to_class = {
    "nlu": NLU_Evaluator,
    "dm": DM_Evaluator,
    "nlg": NLG_Evaluator,
    "preproc": PreprocEvaluator,
    "sa": SA_Evaluator
  }

  return name_to_class[component]

def get_component(model_name: str, component_name: str, prompt: dict) -> Any:
  # Get rule_based
  if model_name == "rule_based":
    if component_name != "dm":
      raise ValueError("Only the dm component has a 'rule-based' option.")
    rule_dm = RuleBasedDM()
    return DM(rule_dm, prompt)
  else: # Initialize LLM
    load_dotenv()
    login_to_hub()
    device = "auto"
    
    # Init component
    model_loader = ModelLoader(model_name, device)

    if component_name == "nlu":
      return NLU(model_loader, prompt)
    elif component_name == "dm":
      return DM(model_loader, prompt)
    elif component_name == "nlg":
      return NLG(model_loader, prompt)
    elif component_name == "preproc":
      return Preproc(model_loader, prompt)
    elif component_name == "sa":
      return SA(model_loader, prompt)
  
  raise ValueError("Invalid configuration model={model_name}, component={component_name}")
    

def eval() -> None:
  """Execute the agent."""
  args = parse_args()
  model_name = args.model
  component_name = args.component
  # Get test set path
  test_set_path = os.path.join(TEST_DIR, component_name + ".json")
  # Get prompt path
  prompt_path = os.path.join(PROMPT_DIR, component_name + ".yaml")
    
  with open(prompt_path, "r", encoding="utf-8") as file:
    prompt = yaml.safe_load(file)

  component = get_component(model_name, component_name, prompt)

  # Get evaluator class based on which component to evaluate
  eval_class = get_evaluator(component_name)
  evaluator = eval_class(component, test_set_path, prompt)

  # Initialize and run evaluator
  results = evaluator.evaluate()
  
  print(results)

if __name__ == "__main__":
  eval()