from copy import deepcopy


def clean_response(response: dict) -> dict:
    """Clean an NLU response for safe merging into the dialogue state."""
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



def update_ds(ds: dict, nlu_response: dict) -> dict:
    """Merge a cleaned NLU response into the dialogue state `ds`."""
    for key, value in nlu_response.items():
        if value is None:
            continue
        # If the incoming value is a dict (commonly 'slots'), merge recursively
        if isinstance(value, dict):
            if key not in ds or not isinstance(ds.get(key), dict):
                ds[key] = {}
            for subk, subv in value.items():
                if subv is None:
                    continue
                ds[key][subk] = subv
        else:
            ds[key] = value
    return ds
