"""
Extensions and utilities for the comp560 package in the nanoGPT project.

This module provides additional functionality to support custom configurations
and extensions without modifying core scripts like train.py, sample.py.
"""

import os


def get_config_file():
    return os.environ.get("NANOGPT_CONFIG", "configurator.py")
