# The functions here are copied from: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
from typing import Optional

import torch


def _multinomial_sample_one_no_sync(probs_sort):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def _logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample_gpt_fast(outputs, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = _logits_to_probs(outputs[0, -1], temperature, top_k)
    idx_next = _multinomial_sample_one_no_sync(probs)
    return idx_next
