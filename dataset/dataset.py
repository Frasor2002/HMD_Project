import pandas as pd

class GameDataset:
  """GameDataset class used to get data to return to the user."""

  def __init__(self, filepath: str  = "dataset/steam_dataset.feather"):
    """Initialize games dataset by reading feather file.
    Args:
      filepath (str): filepath to the feather file.
    """
    
    self.filepath = filepath
    self.df = pd.read_feather(filepath)

  
