from typing import Callable, Sequence, Optional

import torch
import llm_sampler
from tokenizers import Tokenizer


def score_batch(
        input_text: str,
        continuations: Sequence[str],
        tokenizer: Tokenizer,
        forward_func: Callable[[torch.Tensor], torch.Tensor],
        device: Optional[str] = None,
) -> torch.Tensor:
    continuation_ids = [torch.tensor(enc.ids) for enc in tokenizer.encode_batch(continuations)]
    pad_token = tokenizer.token_to_id('<|endoftext|>')
    input_ids = torch.tensor(tokenizer.encode(input_text).ids)
    if device is not None:
        input_ids = input_ids.to(device)
    return llm_sampler.score_batch(
        forward_func=forward_func,
        input_ids=input_ids,
        all_continuation_ids=continuation_ids,
        padding_value=pad_token
    )
