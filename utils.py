import json

def is_serializable(value):
    """
    Check if a value is JSON serializable.
    """
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False
    
def make_serializable(obj):
    """
    Recursively process a data structure to exclude non-serializable keys
    or convert non-serializable objects to strings.
    """
    if isinstance(obj, dict):
        # Process dictionaries: Recursively process each key-value pair
        return {
            key: make_serializable(value)
            for key, value in obj.items()
            if isinstance(value, (str, int, float, list, dict, bool, type(None))) or is_serializable(value)
        }
    elif isinstance(obj, list):
        # Process lists: Recursively process each element
        return [make_serializable(item) for item in obj]
    elif is_serializable(obj):
        # Directly return serializable objects
        return obj
    else:
        # Convert non-serializable objects to strings (or exclude them if needed)
        return str(obj)