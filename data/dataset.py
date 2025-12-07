import pandas as pd
import os

"""Index(['appid', 'name', 'release_date', 'required_age', 'price', 'dlc_count',
  'detailed_description', 'about_the_game', 'short_description',
  'reviews', 'header_image', 'website', 'support_url', 'support_email',
  'windows', 'mac', 'linux', 'metacritic_score', 'metacritic_url',
  'achievements', 'recommendations', 'notes', 'supported_languages',
  'full_audio_languages', 'packages', 'developers', 'publishers',
  'categories', 'genres', 'screenshots', 'movies', 'user_score',
  'score_rank', 'positive', 'negative', 'estimated_owners',
  'average_playtime_forever', 'average_playtime_2weeks',
  'median_playtime_forever', 'median_playtime_2weeks', 'discount',
  'peak_ccu', 'tags', 'pct_pos_total', 'num_reviews_total',
  'pct_pos_recent', 'num_reviews_recent'],
  dtype='object')
"""

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DATA_DIR,"steam_dataset")



class GameDataset:
  """GameDataset class used to get data to return to the user."""

  def __init__(self):
    """Initialize games dataset by reading feather file.
    Args:
      filepath (str): filepath to the feather file.
    """
    # Load dataset
    self.df = pd.read_feather(DATASET_PATH)

  def game_by_title(self, title: str):
    # assume title already normalize
    # output must be always dict {}
     
    match = self.df[self.df['name_normalized'] == title]

    if match.empty:
      match = self.df[self.df['name_normalized'].str.contains(title, na=False)]
      
    if match.empty:
      return None
    
    # Return the most popular result if multiple matches (based on reviews)
    return match.sort_values(by='release_date', ascending=True).iloc[0]
  
  def get_data_by_intent(self):
    pass

  
