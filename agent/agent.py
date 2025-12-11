from agent.dst import DST
from models.model import ModelLoader, LLMTask
from collections import deque
from typing import Deque, Dict
import yaml
import os
import json

class DialogueAgent:
  def __init__(self, model: str, device: str = "cuda", n_exchanges: int = 3) -> None:
    """Initialize dialogue agent.
    Args:
      model (str): model name to load.
      device (str): device where to run the model on.
      n_exchanges (int): number of exchanges to keep in conversation history.
    """
    self.model_name = model
    self.device = device
    self.n_exchanges = n_exchanges

    # Load model
    model_loader = ModelLoader(self.model_name, self.device)

    # Load prompts
    self.system_prompt = self._load_prompt()

    # Instantiate components
    self.preproc = LLMTask(model_loader, self.system_prompt["preproc"]["prompt"])
    self.nlu = LLMTask(model_loader, self.system_prompt["nlu"]["prompt"])
    self.dm = LLMTask(model_loader, self.system_prompt["dm"]["prompt"]["main"])
    self.nlg = LLMTask(model_loader, self.system_prompt["nlg"]["prompt"]["main"])
    self.sa = LLMTask(model_loader, self.system_prompt["sa"]["prompt"])

    # Create Dialogue State Tracker
    self.dst = DST()

    # Conversation history
    max_len = self.n_exchanges * 2
    self.history: Deque[Dict[str, str]] = deque(maxlen=max_len)



  def _load_prompt(self) -> dict:
    """Load a yaml files and get system prompts.
    Returns:
      dict: dict containing the prompts.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_dir = os.path.join(base_dir, "prompt")

    prompts = {}

    for filename in os.listdir(prompt_dir):
      if filename.endswith(".yaml"):
        file_path = os.path.join(prompt_dir, filename)
        key_name = os.path.splitext(filename)[0]
        
        with open(file_path, "r", encoding="utf-8") as file:
          prompts[key_name] = yaml.safe_load(file)
                
    return prompts
  
  def get_review_sa(self, reviews):
    """Given list of reviews return a report of positive and negative."""
    return {"positive": 0, "negative": 0}

  def get_knowledge(self, nba, ds) -> str:
    """Get external knowledge given nba and ds."""
    return ""


  def chat(self, user_input: str) -> str:
    """Chat with the model giving a user input.
    Args:
      user_input (str): user input.
    Returns:
      str: assistant response.
    """
    print(user_input, list(self.history))

    # Keep track if there are multiple intents
    multiple_intents = False

    # Go through the preprocess to split multiple intents
    preproc_out = self.preproc.generate(user_input)
    
    # Convert preproc_out from string to list
    split_input = json.loads(preproc_out)

    # If there are multiple intents the agent will only attend to the last one
    # For better user experience the agent will know when there are multiple intents with a 
    # flag and also add to history other inputs for context
    if len(split_input) > 1:
      multiple_intents = True
      for input in split_input[:-1]:
        self.history.append({"role": "user", "content": input})
    
    # Agent will always attend to last user intent
    nlu_input = split_input[-1]
    
    # Pass through nlu to get intents
    nlu_out = self.nlu.generate(nlu_input, list(self.history))

    print(f"NLU OUT->{nlu_out}")
    # Merge dialogue state and get new state
    self.dst.update_ds(nlu_out)
    ds = self.dst.get_ds()
    print(f"DST OUT->{ds}")


    # Get intent based prompt for dm
    intent_name = self.dst.ds["intent"]
    self.dm.change_system_prompt(self.system_prompt["dm"]["prompt"]["main"] + self.system_prompt["dm"]["prompt"][intent_name])

    # Pass through dm to get next best action
    nba = self.dm.generate(ds, list(self.history))
    print(f"DM OUT->{nba}")

    # TODO Based on intent get external knowledge
    ek = self.get_knowledge(nba, ds)

    # Get intent based prompt for dm
    self.nlg.change_system_prompt(self.system_prompt["nlg"]["prompt"]["main"] + self.system_prompt["nlg"]["prompt"][intent_name])


    # Prepare the nlg input and get natural language output
    nlg_input = f"NBA: {nba}\nDS: {self.dst.get_ds()}\nEK: {ek}\n MI: {multiple_intents}"
    response = self.nlg.generate(nlg_input, list(self.history))

    # Update history
    self.history.append({"role": "user", "content": user_input})
    self.history.append({"role": "assistant", "content": response})

    return response



