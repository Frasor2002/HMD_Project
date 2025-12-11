import pandas as pd
import json
import os
from typing import Any, Optional


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_PATH = os.path.join(DATA_DIR,"steam_dataset.feather")
USER_PROFILE_PATH = os.path.join(DATA_DIR,"mock_user.json")
GLOSSARY_PATH = os.path.join(DATA_DIR,"video_game_glossary.json")


class KnowledgeBase:
  """KnowledgeBase class used to get data to return to the user."""

  def __init__(self):
    """Initialize games dataset by reading feather file.
    Args:
      filepath (str): filepath to the feather file.
    """
    # Load games dataset
    self.df = pd.read_feather(GAMES_PATH)
    # Load user profile
    self.user_profile = self._load_json(USER_PROFILE_PATH)
    self.glossary = self._load_json(GLOSSARY_PATH)
  
  def _load_json(self, path: str) -> Any:
    """Utility to load json files."""
    with open(path, "r", encoding="utf-8") as file:
      return json.load(file)

  def _save_json(self, data:dict, path: str) -> None:
    """Save json file."""
    with open(path, "w", encoding="utf-8") as file:
      json.dump(data, file, indent=2)

  def game_by_title(self, title: str) -> Optional[dict]:
    """Get a game given the title"""
    # assume title already normalize
    # output must be always dict {}
     
    match = self.df[self.df['name_normalized'] == title]

    if match.empty:
      match = self.df[self.df['name_normalized'].str.contains(title, na=False)]
      
    if match.empty:
      return None
    
    result = match.sort_values(by='release_date', ascending=True).iloc[0]
    return result.to_dict()
  

  def get_data_by_intent(self, intent:str, slots: dict) -> dict:
    """Retrieve data based on NLU intent."""
    handlers = {
      "get_game_info": self._handle_get_game_info,
      "discover_game": self._handle_discover_game,
      "compare_games": self._handle_compare_games,
      "get_friend_games": self._handle_get_friend_games,
      "get_term_explained": self._handle_get_term_explained,
      "add_to_wishlist": self._handle_add_wishlist,
      "remove_from_wishlist": self._handle_remove_wishlist,
      "get_wishlist": self._handle_get_wishlist,
    }

    handler = handlers.get(intent)
        
    if handler:
      return handler(slots)
    else:
      return {"error": f"Intent '{intent}' not implemented."}


  def _handle_get_game_info(self, slots):
    title = slots.get('title')
    info_type = slots.get('info', 'summary') # Default to summary
        
    game = self.game_by_title(title)
    if not game:
      return {"response": f"No game called '{title}'."}

    # Map info -> column in database
    mapping = {
      "summary": "short_description",
      "genre": "genres",
      "mode": "categories",
      "required_age": "required_age",
      "price": "price",
    }


    # If querying for platform we need to check for every device column    
    if info_type == "platform":
      platforms = [p for p in ["windows", "mac", "linux"] if game.get(p) is True]
      return {"game": game['name'], "platforms": platforms}
    
    key = mapping.get(info_type)

    val = game.get(key)
    if info_type == "mode":
      val = game.get("categories", [])
      mode_info = {
        "singleplayer": any("single" in c for c in val),
        "multiplayer": any("multi" in c for c in val)
      }
      val = mode_info
    return {"game": game['name'], info_type: val}
        
  
  def _handle_discover_game(self, slots):
    """Filters dataset based on criteria."""
    filtered_df = self.df.copy()

    # Filtering genre
    if slots.get('genre'):
      filtered_df = filtered_df[filtered_df['genres'].str.contains(slots['genre'], case=False, na=False)]

    # Filtering the price as an upper bound
    if slots.get('price'):
      price_limit = float(slots['price'])
      filtered_df = filtered_df[filtered_df['price'] <= price_limit]

    # Filter release year
    if slots.get('release_date'):
      #TODO release_date is a string y-m-d in database i need to check year
      year = str(slots['release_date'])
      filtered_df['release_year'] = filtered_df['release_date'].str[:4]  # extract the year
      filtered_df = filtered_df[filtered_df['release_year'] == year]

    # Filter platform
    if slots.get('platform'):
      plat = slots['platform']
      filtered_df = filtered_df[filtered_df[plat] == True]

    # Filter game mode
    if slots.get('mode'):
      filtered_df = filtered_df[filtered_df['categories'].str.contains(slots['mode'], case=False, na=False)]

    # Filter required age
    if slots.get('required_age'):
      age = int(slots['required_age'])
      filtered_df = filtered_df[filtered_df['required_age'] == age]


    # Filter publisher and developer using the normalized fields
    if slots.get('publisher'):
      filtered_df = filtered_df[filtered_df['publishers_normalized'].str.contains(slots['publisher'], case=False, na=False)]
    if slots.get('developer'):
      filtered_df = filtered_df[filtered_df['developers_normalized'].str.contains(slots['developer'], case=False, na=False)]

    # Sort by price
    results = filtered_df.sort_values(by='price', ascending=False).head(5)

    if results.empty:
      return {}

    return {"proposals": results[['name', 'price', 'release_date']].to_dict()}


  def _handle_compare_games(self, slots):
    t1 = slots.get('title1')
    t2 = slots.get('title2')
    criteria = slots.get('criteria')

    g1 = self.game_by_title(t1)
    g2 = self.game_by_title(t2)

    if not g1 or not g2:
      return {}

    if criteria == "price":
      # TODO study if additional knowledge can be added
      return {"comparison": f"{g1['name']} is ${g1.get('price')} vs {g2['name']} is ${g2.get('price')}"}
    elif criteria == "genre":
      return {"comparison": f"{g1['name']} is {g1.get('genres')} vs {g2['name']} is {g2.get('genres')}"}


  def _handle_get_friend_games(self, slots):
    friend_name = slots.get('name')
        
    # Search in loaded user profile mock
    friends = self.user_profile.get('friends', [])
        
    # Simple match
    for friend in friends:
      if friend['username'].lower() == friend_name.lower():
        return {"games_owned": friend.get('owned', [])}
        
    return {"response": f"Friend '{friend_name}' not found in your friends list."}


  def _handle_get_term_explained(self, slots):
    term = slots.get('term')
        
    # Search for direct match
    definition = self.glossary.get(term)
        
    if not definition:
      # Search harder for matches
      for key, val in self.glossary.items():
        if key.lower() == term.lower():
          return {"term": key, "definition": val}
        return {"response": f"I don't have a definition for '{term}'."}
        
    return {"term": term, "definition": definition}
  

  def _handle_add_wishlist(self, slots):
    title = slots.get('title')
    game = self.game_by_title(title)
        
    if not game:
      return {"response": f"Cannot add '{title}' to wishlist as it was not found in the database."}
        
    # Check if already in wishlist
    current_wishlist = self.user_profile.get('wishlist', [])
    if game['name'] not in current_wishlist:
      self.user_profile['wishlist'].append(game['name'])
      self._save_json(self.user_profile, USER_PROFILE_PATH)
      return {"response": f"Added '{game['name']}' to your wishlist.", "status": "success"}
        
    return {"response": f"'{game['name']}' is already in your wishlist."}


  def _handle_remove_wishlist(self, slots):
    title = slots.get('title')
    current_wishlist = self.user_profile.get('wishlist', [])
    # Check match (case insensitive)
    match = next((g for g in current_wishlist if g.lower() == title.lower()), None)
        
    if match:
      self.user_profile['wishlist'].remove(match)
      self._save_json(self.user_profile, USER_PROFILE_PATH)
      return {"response": f"Removed '{match}' from your wishlist.", "status": "success"}
        
    return {"response": f"'{title}' was not found in your wishlist."}
  

  def _handle_get_wishlist(self, slots):
    wl = self.user_profile.get('wishlist', [])
    if not wl:
        return {"response": "Your wishlist is empty."}
    return {"wishlist": wl}

  def get_reviews(self, game):
    """Using API get up to date reviews on a game"""
    return []


if __name__ == "__main__":
  kb = KnowledgeBase()

  #print(kb.game_by_title("darkwood"))

  #intent = "get_game_info"
  #slots = {"title": "postal 2", "info": "summary"}

  #print(kb.get_data_by_intent(intent, slots))
