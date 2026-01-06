from agent.dst import DST
from models.model import ModelLoader, LLMTask
from data.kb import KnowledgeBase
from collections import deque
from typing import Deque, Dict
import yaml
import os
import re
from agent.preproc import Preproc
from agent.nlu import NLU
from agent.dm import DM, RuleBasedDM
from agent.nlg import NLG
from agent.sa import SA
from typing import Optional
from dotenv import load_dotenv
from models.utils import login_to_hub

class DialogueAgent:
  def __init__(self, model: Dict[str, str], device: str = "cuda", n_exchanges: int = 3) -> None:
    """Initialize dialogue agent.
    Args:
      model (Dict[str, str]): model names to load for each component.
      device (str): device where to run the model on.
      n_exchanges (int): number of exchanges to keep in conversation history.
    """
    self.model_name = model
    self.device = device
    self.n_exchanges = n_exchanges
    
    # Load knowledge base
    self.kb = KnowledgeBase()
    # Load model
    # Model loader cache
    self.loaders = {}
    # Load prompts
    self.system_prompt = self._load_prompt()

    # Instantiate components
    self.preproc = Preproc(self._get_loader("preproc"), self.system_prompt["preproc"])
    
    self.nlu = NLU(self._get_loader("nlu"), self.system_prompt["nlu"])
    
    # dm can be either llm or rule-based
    if self.model_name.get("dm") == "rule_based": dm_loader = RuleBasedDM()
    else: dm_loader = self._get_loader("dm")
    self.dm = DM(dm_loader, self.system_prompt["dm"])

    self.nlg = NLG(self._get_loader("nlg"), self.system_prompt["nlg"])
    self.sa = SA(self._get_loader("sa"), self.system_prompt["sa"])

    # Create Dialogue State Tracker
    self.dst = DST()

    # Conversation history
    max_len = self.n_exchanges * 2
    self.history: Deque[Dict[str, str]] = deque(maxlen=max_len)

  def _get_loader(self, component: str) -> ModelLoader:
    """Given model choices from different components, give the right model for the right component.
    Args:

    """
    default_model = self.model_name.get("default")
    model_name = self.model_name.get(component, default_model)
    if not model_name:
      raise ValueError(f"model_name dict must contain a 'default' key or a key for '{component}'")

    # Load model only it is not already loaded
    if model_name not in self.loaders:
      self.loaders[model_name] = ModelLoader(model_name, self.device)
    return self.loaders[model_name]

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

  def clear_history(self) -> None:
    """Clear conversation history and reset the dialogue state tracker."""
    self.history.clear()
    self.dst.reset()

  def get_review_sa(self, reviews: list) -> dict:
    """Given list of reviews return a report of positive and negative.
    Args:
      reviews (list): list of reviews strings for a given game.
    Returns:
      dict: report of user sentiment on the game.
    """
    report = self.sa.analyze(reviews)
    return report


  def get_knowledge(self, nba: str, ds: dict) -> dict:
    """Get external knowledge given nba and ds.
    Args:
      nba (str): next best action.
      ds (dict): dialogue state,
    Returns:
      dict: external knowledge inside a json object.
    """
    pattern = r'^([a-zA-Z_]\w*)\s*\('
    match = re.match(pattern, nba)
    # Error if dm gave an incorrect output
    if not match:
      return {}
    action_name = match.group(1)
    # If action does not require knowledge, agent does not request it.
    if action_name in ["ask_for", "fallback"]:
      return {}
    
    # Handle different intents
    intent = ds.get("intent")
    slots = ds.get("slots", {})

    # Based on different intents different data is requested
    match intent:
      case "get_game_info":
        title = slots.get("title")
        info = slots.get("info")
        data = self.kb.get_game_info(title, info)
        # If data has reviews, the sa component will be called
        if "review" in data:
          sa = self.get_review_sa(data["review"])
          data["review"] = sa
      case "discover_game":
        data = self.kb.discover_game(**slots)
      case "compare_games":
        data = self.kb.compare_games(**slots)
        if "review" in data:
          # Execute sa if reviews are requested
          for title, review_list in data["review"].items():
            sa = self.get_review_sa(review_list)
            data["review"][title] = sa
      case "get_term_explained":
        data = self.kb.get_term_explained(**slots)
      case "get_friend_games":
        data = self.kb.get_friend_games(**slots)
      case "add_to_wishlist":
        data = self.kb.add_wishlist(**slots)
      case "remove_from_wishlist":
        data = self.kb.remove_wishlist(**slots)
      case "get_wishlist":
        data = self.kb.get_wishlist()
      case _:
        data = {"error": "Invalid intent."}
    
    return data


  def handle_intent(self, nlu_input: str, mi: bool, nlg_tuning: Optional[str] = None) -> str:
    """Handle a single intent given user input.
    Args:
      user_request (str): request on a single intent.
      mi (bool): flag to tell the system if the user requested multiple intents at once.
      nlg_tuning (Optional[str]): additional prompt for the nlg.
    Returns:
      str: nlg output.
    """
    # Go through NLU to extract intents
    nlu_out = self.nlu.generate(nlu_input, list(self.history))
    print(f"Extracted DS -> {nlu_out}")

    # Merge DS and get the updated one
    self.dst.update_ds(nlu_out)
    ds = self.dst.get_ds()
    print(f"DST -> {ds}")

    # Go through DM to get nba
    nba = self.dm.generate(ds)
    print(f"NBA -> {nba}")

    # Get external knowledge if needed
    ek = self.get_knowledge(nba, ds)
    # If error in request, action becomes fallback()
    if "error" in ek:
      print(f"Knowledge error: {ek['error']}")
      nba = "fallback()"
      ek = None

    nlg_out = self.nlg.generate(nba, ds, ek, mi, nlg_tuning)

    return nlg_out

  def handle_two_intents(self, split_input: list) -> str:
    responses = []
    
    add_tunings = ["multiresponse1", "multiresponse2"]

    for i, sub_input in enumerate(split_input):
      additional_tuning = add_tunings[i]

      nlg_out = self.handle_intent(sub_input, mi=False, nlg_tuning=additional_tuning)
      responses.append(nlg_out)
                
      # Give context to next request
      self.history.append({"role": "user", "content": sub_input})
      self.history.append({"role": "assistant", "content": nlg_out})

    # Combine outputs      
    return " ".join(responses)


  def chat(self, user_input: str) -> str:
    """Chat with the model giving a user input.
    Args:
      user_input (str): user input.
    Returns:
      str: assistant response.
    """
    # Keep track if there are multiple intents
    multiple_intents = False

    # Go through the preprocess to split input based on intents
    split_input = self.preproc.generate(user_input)
    print(f"SPLIT -> {split_input}")

    # Based on intent number the agent has different behaviour
    intent_number = len(split_input)
    if intent_number == 2:
      return self.handle_two_intents(split_input)

    if intent_number > 2:
      multiple_intents = True
      for input in split_input[:-1]:
        self.history.append({"role": "user", "content": input})
    
    # Agent will always attend to last user intent
    nlu_input = split_input[-1]
  
    response = self.handle_intent(nlu_input, mi=multiple_intents)

    # Update history
    self.history.append({"role": "user", "content": nlu_input})
    self.history.append({"role": "assistant", "content": response})

    return response
  

def load_agent() -> DialogueAgent:
  """Load the agent with the recommended configuration."""
  # Load hf 
  load_dotenv()
  login_to_hub()

  device = "auto"
  n_exchanges = 2

  # Models used for every component
  model = {
    "default": "qwen3",
    "preproc": "qwen3",
    "nlu": "qwen3",
    "dm": "rule_based",
    "nlg": "llama3",
    "sa": "qwen3"
  }

  dialogue_agent = DialogueAgent(model, device, n_exchanges)
  return dialogue_agent

