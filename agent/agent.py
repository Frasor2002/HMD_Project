from agent.dst import DST
from models.model import ModelLoader, LLMTask
import yaml

class DialogueAgent:
  def __init__(self, model: str, device: str, n_exchanges: int):
    self.model_name = model
    self.device = device
    self.n_exchanges = n_exchanges

    # Load model
    model_loader = ModelLoader(self.model_name, self.device)

    # Load prompt
    self.task_prompt = self._load_prompt("prompt/prompt.yaml")

    # Instantiate components
    self.nlu = LLMTask(model_loader, self.task_prompt["nlu"], self.n_exchanges)
    self.dm = LLMTask(model_loader, self.task_prompt["dm"], self.n_exchanges)
    self.nlg = LLMTask(model_loader, self.task_prompt["nlg"], self.n_exchanges)

    # Create Dialogue State Tracker
    self.dst = DST()



  def _load_prompt(self, prompt_path : str) -> str:
    with open(prompt_path, "r") as file:
      task_prompt = yaml.safe_load(file)
    return task_prompt
  

  def chat(self, user_input: str) -> str:

    nlu_out = self.nlu.generate(user_input)
    print(f"NLU OUT->{nlu_out}")
    self.dst.update_ds(nlu_out)

    ds = self.dst.get_ds()
    print(f"DST OUT->{ds}")
    dm_out = self.dm.generate(ds)
    print(f"DM OUT->{dm_out}")

    nlg_input = f"NBA: {dm_out}\nDS: {self.dst.get_ds()}"
    response = self.nlg.generate(nlg_input)

    return response



