from agent.dst import DST
from models.model import ModelLoader, LLMTask
from data.kb import KnowledgeBase
from collections import deque
from typing import Deque, Dict, Optional, Union
import yaml
import os
import json
import re
from agent.utils import RuleBasedDM, validate_preproc, validate_nlu, validate_dm, validate_sa

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
    self.preproc = LLMTask(self._get_loader("preproc"), self.system_prompt["preproc"]["prompt"])
    self.nlu = LLMTask(self._get_loader("nlu"), self.system_prompt["nlu"]["prompt"])
    # dm can be either llm or rule-based
    if self.model_name.get("dm") == "rule_based":
      self.dm = RuleBasedDM()
    else:
      self.dm = LLMTask(self._get_loader("dm"), self.system_prompt["dm"]["prompt"]["main"])
    self.nlg = LLMTask(self._get_loader("nlg"), self.system_prompt["nlg"]["prompt"]["main"])
    self.sa = LLMTask(self._get_loader("sa"), self.system_prompt["sa"]["prompt"])

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
  
  def set_intent_based_prompt(self, component: str, intent_name: str) -> None:
    """For a specific component set an intent-based prompt.
    Args:
      component (str): component name of which to change the prompt.
      intent_name (str): current intent name.
    """
    match component:
      case "dm":
        self.dm.change_system_prompt(self.system_prompt[component]["prompt"]["main"] + self.system_prompt[component]["prompt"][intent_name])
      case "nlg":
        self.nlg.change_system_prompt(self.system_prompt[component]["prompt"]["main"] + self.system_prompt[component]["prompt"][intent_name])
      case _:
        raise ValueError("Component name not valid.")

    

  def get_review_sa(self, reviews: list) -> dict:
    """Given list of reviews return a report of positive and negative.
    Args:
      reviews (list): list of reviews strings for a given game.
    Returns:
      dict: report of user sentiment on the game.
    """
    report = {"positive": 0, "negative": 0, "neutral": 0}
    for review in reviews:
      raw_label = self.sa.generate(review)
      label = validate_sa(raw_label)
      report[label] += 1

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
        # Default info is summary if not specified
        title = slots.get("title")
        info = slots.get("info", "summary")
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

  def handle_intent(self, nlu_input: str, mi: bool) -> str:
    """Handle a single intent given user input.
    Args:
      user_request (str): request on a single intent.
      mi (bool): flag to tell the system if the user requested multiple intents at once.
    Returns:
      str: nlg output.
    """
    # Go through NLU to extract intents
    raw_nlu_out = self.nlu.generate(nlu_input, list(self.history))
    nlu_out = validate_nlu(raw_nlu_out)
    print(f"Extracted DS -> {nlu_out}")

    # Merge DS and get the updated one
    self.dst.update_ds(nlu_out)
    ds = self.dst.get_ds()
    print(f"DST -> {ds}")

    # Set intent-based prompt for dm
    intent_name = self.dst.ds["intent"]
    self.set_intent_based_prompt("dm", intent_name)

    # Go through DM to get nba
    raw_nba = self.dm.generate(ds)
    nba = validate_dm(raw_nba)
    print(f"NBA -> {nba}")

    # Get external knowledge if needed
    ek = self.get_knowledge(nba, self.dst.ds)
    # If error in request, action becomes fallback()
    if "error" in ek:
      print(f"Knowledge error: {ek["error"]}")
      nba = "fallback()"
      ek = None

    # Set intent-based prompt for nlg
    self.set_intent_based_prompt("nlg", intent_name)

    # Prepare the nlg input and get natural language output
    nlg_input = f"NBA: {nba}\nDS: {ds}\nEK: {ek}\n MI: {mi}"
    print(f"NLG Input -> {nlg_input}")
    nlg_out = self.nlg.generate(nlg_input)

    return nlg_out

  def handle_two_intents(self, split_input: list) -> str:
    responses = []
            
    for sub_input in split_input:
      nlg_out = self.handle_intent(sub_input, mi=False)
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
    raw_preproc_out = self.preproc.generate(user_input)
    split_input = validate_preproc(raw_preproc_out, user_input)
    print(f"SPLITTED -> {split_input}")

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
  



