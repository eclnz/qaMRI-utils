import re

def sanitize_string(input_string: str, replacement: str = "_") -> str:
    """
    Sanitizes a string by replacing spaces and non-alphanumeric characters.
    
    Args:
        input_string (str): The string to sanitize.
        replacement (str): The character to replace spaces and invalid characters with (default: "_").
    
    Returns:
        str: The sanitized string.
    """
    # Replace spaces with the replacement character
    sanitized = input_string.replace(" ", replacement)
    
    # Remove or replace all non-alphanumeric characters except for underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", replacement, sanitized)
    
    # Remove consecutive replacement characters (e.g., "__")
    sanitized = re.sub(rf"{re.escape(replacement)}+", replacement, sanitized)
    
    # Strip leading and trailing replacement characters
    sanitized = sanitized.strip(replacement)
    
    return sanitized