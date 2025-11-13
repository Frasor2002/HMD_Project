import yaml
from models import LLMTask, ModelLoader

class NLU:
  """Natural language understanding component."""
  def __init__(
    self, 
    model: ModelLoader, 
    prompt_path: str = "prompt/nlu_prompt.yaml",
    n_exchanges: int = 2
    ):

    with open(prompt_path, "r") as file:
      task_prompt = yaml.safe_load(file)["nlu"]

    self.nlu_model = LLMTask(model, task_prompt, n_exchanges)
  
  def extract_intent_slots(self, prompt: str):
    """Given a natural language prompt get intents and slots in a json format."""

    return self.nlu_model.generate(prompt)