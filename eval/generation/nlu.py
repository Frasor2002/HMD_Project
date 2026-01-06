import json
import itertools
from string import Formatter
from typing import Callable, Optional
import os

GEN_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(GEN_DIR)
DATASET_PATH = os.path.join(EVAL_DIR, "test_set", "nlu.json")

# Templates for the data (generated from llm)
intent_templates = {
  "get_game_info": ["I want to know more about {title}", "Search for {title}", "What is the {info} of {title}?"],
  "discover_game": ["Find me some {genre} games", "I want to play a game on {platform}", "Show me games made by {developer}", "Are there any games from {release_year}?", "I am looking for {mode} games", "Show me {genre} games playable on {platform}", "I want a {mode} game released in {release_year}", "Do you know any games similar to {similar_title} for {platform}?", "List games by {publisher} that are {genre}","Search for {genre} games on {platform} that cost less than {price}", "Find games like {similar_title} that are for age {required_age} and for {platform}"],
  "compare_games": ["What is the difference between {title1} and {title2}?","Compare the {criteria} of {title1} and {title2}","Tell me about the {criteria} differences between {title1} and {title2}"],
  "get_friend_games": ["What is {name} playing?","Show me the games {name} owns","Does {name} have any recommendations?","List games owned by {name}"],
  "get_term_explained": ["What does {term} mean?","Explain the term {term}","Define {term} in the context of gaming","What is {term}?"],
  "add_to_wishlist": ["Add {title} to my wishlist","Save {title} in my list", "I want to add {title} to my saved games"],
  "remove_from_wishlist": ["Remove {title} from my wishlist","Delete {title} from my list","Take {title} off my saved games"],
  "get_wishlist": ["Show me my wishlist","What games have I saved?","List my wishlist items","Open my list"],
  "out_of_domain": ["What is the weather like?","Book a table for two","How do I cook pasta?","Who is the president?","Navigate to home"]
}

# Slot values for generation
TITLES = ["Terraria", "Metal Gear Solid", "Assassin Creed 2", "The Division", "Far Cry 5", "Rust", "Portal"]
slot_values = {
  # get_game_info slots
  "title": TITLES,
  "info": ["summary", "genre", "mode", "platform", "required_age", "publisher", "developer", "review", "price"],
  # discover_game slots
  "genre": ["action", "adventure", "casual", "early access", "education", "free to play", "game development", "gore", "indie", "massively multiplayer", "movie", "rpg", "racing", "simulation", "sports", "strategy", "violent"],
  "price": [0, 5, 10, 15, 40],
  "release_year": [2007, 2010, 2015, 2022, 2025],
  "platform": ["windows", "mac", "linux"],
  "mode": ["singleplayer", "multiplayer"],
  "similar_title": TITLES,
  "required_age": [0,16,18,21],
  "publisher": ["Ubisoft", "Bethesda", "Activision"],
  "developer": ["Ubisoft", "Bethesda", "Activision"],
  # compare_games slots
  "title1": TITLES,
  "title2": TITLES,
  "criteria": ["price", "review", "genre"],
  # get_friend_names
  "name": ["Alex", "FieryGamer", "FlaireBurn99", "Mark"],
  # get_term_explained
  "term": ["RPG", "adventure game", "level", "AAA", "remake", "rush"]
}

intent_schemas = {
  "get_game_info": [
    "title", 
    "info"
  ],
  "discover_game": [
    "genre", 
    "price", 
    "release_year", 
    "platform", 
    "mode", 
    "similar_title", 
    "required_age", 
    "publisher", 
    "developer"
  ],
  "compare_games": [
    "title1", 
    "title2", 
    "criteria"
  ],
  "get_friend_games": [
    "name"
  ],
  "get_term_explained": [
    "term"
  ],
  "add_to_wishlist": [
    "title"
  ],
  "remove_from_wishlist": [
    "title"
  ],
  "get_wishlist": [],
  "out_of_domain": []
}

def generate_intent_data(intent_name: str, templates: list, slot_values: dict, schema: list, constraints: Optional[Callable] = None) -> list:
  """Generate data for a single intent by adapting to the missing slots in the template.
  Args:
    intent_name (str): name of the intent.
    templates (list): list of templates.
    slot_values (dict): possible values for every slot.
    schema (list): list of slots for current intent to init to None missing ones.
    constraint (Optional[Callable]): constraints for the slot values of the slots in the intent.
  Returns:
    list: list of generated samples.
  """
  data = []

  # Iterate on every template for that intent
  for template in templates:
    # Get missing slots to fill in
    required_slots = [fn for _, fn, _, _ in Formatter().parse(template) if fn is not None]

    # List of values for the required slots
    value_lists = []
    for slot in required_slots:
      value_lists.append(slot_values[slot])
    
    for combination in itertools.product(*value_lists):
      slot_data = dict(zip(required_slots, combination))

      full_slots = {slot: slot_data.get(slot, None) for slot in schema}
      # If constraint not followed, the sample is skipped
      if constraints and not constraints(full_slots):
        continue
      
      # Build and save sample
      sample = {
        "utterance": template.format(**slot_data),
        "annotation": {
          "intent": intent_name,
          "slots": full_slots
        }
      }
      data.append(sample)

  return data




def generate_test_set(print_stats : bool = False) -> list:
  """Generate the entire test set for the nlu component."""

  test_set = []

  for intent, templates in intent_templates.items():
    if intent == "compare_games":
      constraints = lambda slots: (slots.get("title1") is None or slots.get("title2") is None or slots["title1"] != slots["title2"])
    else:
      constraints = None
    schema = intent_schemas[intent]
    intent_data = generate_intent_data(intent, templates, slot_values, schema, constraints)
    
    if print_stats:
      print(f"For intent {intent} {len(intent_data)} samples.")
    test_set.extend(intent_data)

  if print_stats:
    print(f"Total samples {len(test_set)}")
  
  return test_set

if __name__ == "__main__":
  test_set = generate_test_set(True)

  # Save data in dataset file nlu.json
  with open(DATASET_PATH, "w", encoding="utf-8") as file:
    json.dump(test_set, file, indent=4)
