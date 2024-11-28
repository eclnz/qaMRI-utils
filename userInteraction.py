from typing import List, Optional, Any

def prompt_user(prompt: str, default: Optional[str] = None, options: Optional[List[str]] = None, 
                allow_skip: bool = False, parse_type: Optional[type] = str) -> Any:
    """
    Prompt the user for input with options and default values.

    Args:
        prompt (str): The prompt message.
        default (Optional[str]): Default value if no input is given.
        options (Optional[List[str]]): List of valid options.
        allow_skip (bool): Whether the user can skip the input.
        parse_type (Optional[type]): Type to parse the input into.

    Returns:
        Any: The user input parsed to the specified type.
    """
    while True:
        # Build the full prompt message
        msg = f"{prompt} "
        if options:
            msg += f"Options: {', '.join(options)}. "
        if default is not None:
            msg += f"(default: {default}) "

        # Get user input
        response = input(msg).strip()
        if not response and default is not None:
            # Use parse_type only if it's provided, otherwise return the default as-is
            return parse_type(default) if parse_type else default

        if options and response not in options:
            print("Invalid option. Please choose from the available options.")
            continue

        try:
            return parse_type(response) if parse_type else response
        except ValueError:
            print(f"Invalid input. Please enter a valid {parse_type.__name__}.")