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

def calc_flops_achieved(flops_per_iter, dt):
    return flops_per_iter * (1.0/dt) if dt > 0 else 0.0  # per second

