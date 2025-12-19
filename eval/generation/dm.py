import json
import itertools
from typing import Callable, Optional
import os

GEN_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(GEN_DIR)
DATASET_PATH = os.path.join(EVAL_DIR, "test_set", "dm.json")

TITLES = [None, "Terraria", "Metal Gear Solid", "Assassin Creed 2", "The Division", "Far Cry 5", "Rust", "Portal"]
slot_values = {
  # get_game_info slots
  "title": TITLES,
  "info": [None, "summary", "genre", "mode", "platform", "required_age", "publisher", "developer", "review", "price"],
  # discover_game slots
  "genre": [None, "action", "adventure", "casual", "early access", "education", "free to play", "game development", "gore", "indie", "massively multiplayer", "movie", "rpg", "racing", "simulation", "sports", "strategy", "violent"],
  "price": [None, 0, 5, 10, 15, 40],
  "release_year": [None, 2007, 2010, 2015, 2022, 2025],
  "platform": [None, "windows", "mac", "linux"],
  "mode": [None, "singleplayer", "multiplayer"],
  "similar_title": TITLES,
  "required_age": [None, 0,16,18,21],
  "publisher": [None, "Ubisoft", "Bethesda", "Activision"],
  "developer": [None, "Ubisoft", "Bethesda", "Activision"],
  # compare_games slots
  "title1": TITLES,
  "title2": TITLES,
  "criteria": [None, "price", "review", "genre"],
  # get_friend_names
  "name": [None, "Alex", "FieryGamer", "FlaireBurn99", "Mark"],
  # get_term_explained
  "term": [None, "RPG", "adventure game", "level", "AAA", "remake", "rush"]
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

def get_action(intent: str, slots: dict) -> str:
  """Given a ds return the action annotation.
  Args:
    intent (str): dialogue state intent.
    slots (dict): dict of dialogue state.
  Returns:
    str: next best action annotation based on some rules.
  """
  # Get filled slots
  filled_slots = {k: v for k, v in slots.items() if v is not None}

  match intent:
    case "get_game_info":
      if slots.get("title") is None:
        action = "ask_for(title)"
      elif slots.get("info") is None:
        action = "ask_for(info)"
      else:
        info_val = slots.get("info")
        action = f"give_info(title, {info_val})"
    
    case "discover_game":
      if len(filled_slots) < 1:
        action = "ask_for(genre)"
      else:
        action = f"propose_game({', '.join(filled_slots.keys())})"
    
    case "compare_games":
      if slots.get("title1") is None:
        action = "ask_for(title1)"
      elif slots.get("title2") is None:
        action = "ask_for(title2)"
      elif slots.get("criteria") is None:
        action = "ask_for(criteria)"
      else:
        action = f"give_comparison({', '.join(filled_slots.keys())})"
    
    case "get_friend_games":
      if slots.get("name") is None:
        action = "ask_for(name)"
      else:
        action = "give_friend_games(name)"
    
    case "get_term_explained":
      if slots.get("term") is None:
        action = "ask_for(term)"
      else:
        action = "explain_term(term)"
    
    case "add_to_wishlist":
      if slots.get("title") is None:
        action = "ask_for(title)"
      else:
        action = "add_game(title)"
    
    case "remove_from_wishlist":
      if slots.get("title") is None:
        action = "ask_for(title)"
      else:
        action = "remove_game(title)"
    
    case "get_wishlist":
      action = "give_wishlist()"
    case "out_of_domain":
      action = "fallback()"
    case _:
      action = "fallback()"


  return action

def generate_test_set(intent_schemas: dict, slot_values: dict, print_stats: bool = False) -> list:
  """Generate test set given schemas of intents and values.
  Args:
    intent_schemas (dict): schemas of intents.
    slot_values (dict): values to use to fill in the slots.
  Returns:
    list: list of test set samples.
  """
  test_set = []
  stats = {intent: 0 for intent in intent_schemas}

  for intent, slots_name_list in intent_schemas.items():
    # If there are no slots for current intent
    if len(slots_name_list) == 0:
      nba = get_action(intent, {})
      # Add sample
      stats[intent] += 1
      test_set.append(
        {
          "ds": {
            "intent": intent,
            "slots": {}
          },
          "annotation": nba
        }
      )
      continue # Skip code for intents with slots

    # discover_game has too many combination and often the user only provides maximum 3 slots
    if intent=="discover_game":
      seen_states = set()
      for active_slots in itertools.combinations(slots_name_list, 3):
        active_values_list = [slot_values[s] for s in active_slots]
        for combination in itertools.product(*active_values_list):
          slot_data = {s: None for s in slots_name_list}     
          for s_name, s_val in zip(active_slots, combination):
            slot_data[s_name] = s_val

          state_hash = tuple(slot_data[s] for s in slots_name_list)
                    
          if state_hash in seen_states:
            continue
          seen_states.add(state_hash)

          nba = get_action(intent, slot_data)
          test_set.append({
            "ds": {
              "intent": intent,
              "slots": slot_data
            },
            "annotation": nba
          })
          stats[intent] += 1
      continue

    values_list = []
    for slot_name in slots_name_list:
      # get slot values
      values = slot_values[slot_name]
      values_list.append(values)
    
    for combination in itertools.product(*values_list):
      slot_data = dict(zip(slots_name_list, combination))

      # Constraint for compare games
      if intent == "compare_games":
        t1 = slot_data.get("title1")
        t2 = slot_data.get("title2")
        if t1 is not None and t2 is not None and t1 == t2:
          continue

      nba = get_action(intent, slot_data)
      # Create and save samples

      sample = {
        "ds": {
          "intent": intent,
          "slots": slot_data
        },
        "annotation": nba
      }
      stats[intent] += 1
      test_set.append(sample)

  if print_stats:
    print(stats)
  return test_set

if __name__ == "__main__":
  test_set = generate_test_set(intent_schemas, slot_values, True)

  # Save data in dataset file nlu.json
  with open(DATASET_PATH, "w", encoding="utf-8") as file:
    json.dump(test_set, file, indent=4)