from copy import deepcopy
import json

# TODO clean and add methods for class

class DST:
  def __init__(self):
    """Initialize Dialogue State Tracker module."""

    # Empty dialogue state at the start
    self.ds = {
      "intent": None,
      "slots": {}
    }
  
  def get_ds(self) -> str:
    return json.dumps(self.ds)
  
  @staticmethod
  def _clean_response(response: dict) -> dict:
    """Clean an NLU response for safe merging into the dialogue state.
    Args:
      response (str): NLU response to clean.
    Returns:
      dict: cleaned response.
    """
    def _clean(obj):
      if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
          # Normalize explicit textual nulls
          if isinstance(v, str) and v.strip().lower() == "null":
            v = None
          if v is None:
            continue
          if isinstance(v, dict):
            cleaned = _clean(v)
            if cleaned:
              out[k] = cleaned
          else:
            out[k] = v
        return out
      else:
        return obj

    return _clean(deepcopy(response))


  def update_ds(self, nlu_response: dict) -> None:
    """Merge a cleaned NLU response into the dialogue state `ds`."""

    # Convert response in dict
    nlu_response = json.loads(nlu_response)
    nlu_response = self._clean_response(nlu_response)
    
    for key, value in nlu_response.items():
      if value is None:
        continue
      # If the incoming value is a dict (commonly 'slots'), merge recursively
      if isinstance(value, dict):
        if key not in self.ds or not isinstance(self.ds.get(key), dict):
          self.ds[key] = {}
        for subk, subv in value.items():
          if subv is None:
            continue
          self.ds[key][subk] = subv
      else:
        self.ds[key] = value
