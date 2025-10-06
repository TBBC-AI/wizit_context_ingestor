import re


def has_invalid_file_name_format(file_name):
    """Check if file name has special characters or spaces instead of underscores"""
    # Check for spaces
    if " " in file_name:
        return True

    # Check for special characters (anything that's not alphanumeric, underscore, dash, or dot)
    if re.search(r"[^a-zA-Z0-9_.-]", file_name):
        return True
    return False
