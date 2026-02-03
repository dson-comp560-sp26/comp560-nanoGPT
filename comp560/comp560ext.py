"""
Extensions and utilities for the comp560 package in the nanoGPT project.

This module provides additional functionality to support custom configurations
and extensions without modifying core scripts like train.py, sample.py.
"""

import os
import torch
from torch.nn import functional as F

config = {} # Should be overwritten by train.py or sample.py

def get_config_file():
    return os.environ.get("NANOGPT_CONFIG", "configurator.py")


def print_config():
    print(f'comp560ext.config:\n{config}\n-----------------')

def calc_flops_achieved(flops_per_iter, dt):
    return flops_per_iter * (1.0/dt) if dt > 0 else 0.0  # per second

def prepare_stop_token(config_dict, encode):
    """
    Encodes the stop_token string into a token ID using the provided encode function.
    config_dict: A dictionary (e.g., globals()) to look up 'stop_token'.
    Returns the first token ID or None.
    """
    stop_token = config_dict.get('stop_token', None)
    if stop_token is not None:
        stop_ids = encode(stop_token)
        if len(stop_ids) > 0:
            return stop_ids[0]
    return None

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    
    Args:
        stop_token (int, optional): If provided, generation stops when this token is generated.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        
        # stop response when stop_token is generated
        if stop_token is not None and idx_next.item() == stop_token:
            break

    return idx

