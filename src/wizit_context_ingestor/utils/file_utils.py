import re


def validate_file_name_format(file_name):
    """Check if file name has special characters or spaces instead of underscores"""
    # Check for special characters (anything that's not alphanumeric, underscore, dash, dot, slash, or backslash)
    if re.search(r"[^a-zA-Z0-9_.\-/\\]", file_name) is None:
        return True
    else:
        raise ValueError(
            "Invalid file name format, do not provide special characters or spaces (instead use underscores or hyphens)"
        )
