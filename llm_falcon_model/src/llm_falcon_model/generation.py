from copy import deepcopy
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

import llm_sampler
from tokenizers import Tokenizer


def generate(
        input_text: str,
        tokenizer: Tokenizer,
        forward_func: nn.Module,
        max_new_tokens: int,
        temperature: float,
        warp_top_k: Optional[int] = 10,
        device: Optional[str] = None,
) -> str:
    input_ids = tokenizer.encode(input_text).ids
    batch = torch.tensor(input_ids).unsqueeze(0)
    if device:
        batch = batch.to(device)
    result_tokens = deepcopy(input_ids)
    gen = llm_sampler.sample(
        forward_func=forward_func,
        input_ids=batch,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        warp_top_k=warp_top_k
    )
    for tok in tqdm(gen, total=max_new_tokens):
        result_tokens.append(tok.detach().cpu().item())
    return tokenizer.decode(result_tokens)
