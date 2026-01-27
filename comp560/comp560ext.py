"""
Extensions and utilities for the comp560 package in the nanoGPT project.

This module provides additional functionality to support custom configurations
and extensions without modifying core scripts like train.py, sample.py.
"""

import os

config = {} # Should be overwritten by train.py or sample.py

def get_config_file():
    return os.environ.get("NANOGPT_CONFIG", "configurator.py")


def print_config():
    print(f'comp560ext.config:\n{config}\n-----------------')

def logger_print(*args, log_file="output.log", **kwargs):
    """
    Prints content to the console and simultaneously appends it to a log file.
    
    This function acts as a wrapper for the standard Python print() function. 
    It captures all positional and keyword arguments, formats them, and 
    writes them to a specified file.

    Args:
        *args: Variable length argument list of objects to print.
        log_file (str): The path to the file where output should be logged. 
            Defaults to "output.log".
        **kwargs: Arbitrary keyword arguments passed directly to the 
            builtin print() function (e.g., sep, end, flush).

    Returns:
        None

    Acknowledgment:
        This function was authored with the assistance of Gemini, an AI by Google.
    """
    # 1. Format the message exactly as print() would
    # Using a join on string conversions of args
    sep = kwargs.get('sep', ' ')
    message = sep.join(str(arg) for arg in args)
    
    # 2. Print to the console (original behavior)
    print(*args, **kwargs)

    # 3. Append to the log file
    with open(log_file, "a", encoding="utf-8") as f:
        # We use 'end' from kwargs if it exists, otherwise default to newline
        f.write(message + kwargs.get('end', '\n'))

# --- Usage ---
logger_print("System initialized.")
logger_print("Data processed:", 42, sep=" -> ")