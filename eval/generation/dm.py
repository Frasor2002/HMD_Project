import json
import itertools
import os
from agent.utils import get_action

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



def get_strided_combinations(iter_data: itertools.product, limit: int) -> list:
  """Skip some combations according to a step to limit samples maximizing variety.
  Args:
    iter_data: combo.
    limit: limit of samples per combo.
  Returns:
    list: new combos to iterate on.
  """
  all_combos = list(iter_data)
  total = len(all_combos)
  
  if total == 0:
    return []
  
  if total <= limit:
    return all_combos
  
  step = total / limit
  indices = [int(i * step) for i in range(limit)]
  
  unique_indices = sorted(list(set(indices)))
  
  return [all_combos[i] for i in unique_indices]

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
    # discover_game
    elif intent=="discover_game":
      # Reduce the amount of values used for generating a single combination
      max_samples_per_combo = 10
      
      for r in range(1, 4): # From 1 to 3 active slots
        slot_combinations = list(itertools.combinations(slots_name_list, r))

        for filled_slots in slot_combinations:
          # get values for currently active slots
          filled_values_list = [slot_values[s] for s in filled_slots]
          value_combo = itertools.product(*filled_values_list)
          value_combo = get_strided_combinations(value_combo, max_samples_per_combo)

          for values in value_combo:
            slot_data = {s: None for s in slots_name_list}     
            for s_name, s_val in zip(filled_slots, values):
              slot_data[s_name] = s_val
            nba = get_action(intent, slot_data)
            test_set.append({
              "ds": {
                "intent": intent,
                "slots": slot_data
              },
              "annotation": nba
            })
            stats[intent] += 1
    else:
      # For every other intent
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
          if t1 is not None and t2 is not None and t1 == t2: # Skip if constraint not respected
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
    for intent, number in stats.items():
      print(f"For intent {intent} {number} samples")
    print(f"Total samples {len(test_set)}")
  return test_set

if __name__ == "__main__":
  test_set = generate_test_set(intent_schemas, slot_values, True)

  # Save data in dataset file nlu.json
  with open(DATASET_PATH, "w", encoding="utf-8") as file:
    json.dump(test_set, file, indent=4)