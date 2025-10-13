# Imports
import re
import numpy as np


def map_first_digit_to_value(input_string, mapping_dict, default_value="Unknown"):
    """
    Maps the first digit of a string to a specific value based on a provided dictionary.

    Args:
        input_string (str): The string whose first digit you want to map.
        mapping_dict (dict): A dictionary where keys are the first digits (as strings)
                             and values are the corresponding mapped values.
        default_value (any): The value to return if the first digit is not found
                             in the mapping_dict. Defaults to "Unknown".

    Returns:
        any: The mapped value or the default_value if no match is found.
    """
    # Cast to string and strip whitespace
    input_string = str(input_string).strip()

    # ADD THIS CHECK: If the string is empty AFTER stripping, return the default value
    if not input_string:
        return default_value
    
    # The original check 'if not input_string:' is no longer needed at the top,
    # as the two lines above now cover both empty and space-only inputs robustly.

    first_digit = input_string[0] # This line is now safe
    
    return mapping_dict.get(first_digit, default_value)
   

def get_credit_card_network(card_number):
    """
    Determines the credit card network based on the card number's BIN (Bank Identification Number).

    Args:
        card_number (str): The credit card number as a string.
                           It will remove spaces and hyphens automatically.

    Returns:
        str: The name of the credit card network (e.g., "Visa", "Mastercard", "American Express", etc.)
             or "Unknown" if the network cannot be determined.
    """
    # Ensure card_number is a string at the very beginning of the function
    # This handles cases where it might be an int, float, or NaN from the DataFrame
    card_number_str = str(card_number)
    
    # Handle explicit NaN values after conversion to string
    if card_number_str.lower() == 'nan':
        return "Unknown"

    # Remove any spaces or hyphens from the card number string
    clean_number = re.sub(r'[\s\-]', '', card_number_str)

    # Dictionary mapping regex patterns to credit card networks
    card_patterns = {
        "Visa": r"^4\d+",
        "Mastercard": r"^(5[1-5]|222[1-9]|22[3-9]|2[3-6]|27[01]|2720)\d+",
        "American Express": r"^3[47]\d+",
        "Discover": r"^(6011|65|64[4-9]|622(12[6-9]|1[3-9][0-9]|[2-8][0-9]{2}|9[01][0-9]|92[0-5]))\d+",
        "Diners Club": r"^3(?:0[0-5]|[689])\d+",
        "JCB": r"^(352[8-9]|35[3-8][0-9])\d+",
        "UnionPay": r"^(62|88)\d+",
        "Maestro": r"^(50|5[6-9]|6[0-9])\d+"
    }

    for network, pattern in card_patterns.items():
        if re.match(pattern, clean_number): # Use clean_number here
            return network
    
    return "Unknown"