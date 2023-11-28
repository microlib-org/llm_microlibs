from typing import Callable, Sequence, Optional, List

import torch
from torch import nn


def _top_k_logits_warper(
        scores: torch.FloatTensor,
        top_k: int,
        filter_value: float = -float("Inf")
) -> torch.FloatTensor:
    top_k = min(top_k, scores.size(-1))
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


@torch.no_grad()
def sample(
        forward_func: Callable[[torch.Tensor], torch.Tensor],
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.,
        warp_top_k: Optional[int] = 10,
):
    for i in range(max_new_tokens):
        outputs = forward_func(input_ids)
        logits = outputs[:, -1, :]
        scores = _top_k_logits_warper(logits, top_k=warp_top_k) if warp_top_k > 0 else logits
        scores = scores / temperature if abs(temperature) > 1e-7 else scores
        probs = nn.functional.softmax(scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        yield next_tokens
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=(-1))


@torch.no_grad()
def _sample_score(
        forward_func: Callable[[torch.Tensor], torch.Tensor],
        input_ids: torch.Tensor,
        continuation_ids: torch.Tensor,
):
    for i in range(continuation_ids.shape[1]):
        outputs = forward_func(input_ids)
        next_token_logits = outputs[:, -1, :]
        next_tokens = continuation_ids[:, -1]
        yield continuation_ids[:, i].cpu(), next_token_logits.cpu()
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=(-1))


@torch.no_grad()
def score_batch(
        forward_func: Callable[[torch.Tensor], torch.Tensor],
        input_ids: torch.Tensor,
        all_continuation_ids: Sequence[torch.Tensor],
        padding_value: int,
):
    if len(input_ids.shape) == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids.squeeze(0)
    intermediary_continuations = []
    for c_ids in all_continuation_ids:
        c_ids = c_ids.to(input_ids.device)
        intermediary_continuations.append(torch.cat((input_ids, c_ids)))
    batch = torch.nn.utils.rnn.pad_sequence(intermediary_continuations, padding_value=padding_value, batch_first=True)
    batch = batch.to(input_ids.device)
    logits = forward_func(batch)
    lengths = [len(c) for c in all_continuation_ids]
    start_idx = -max(lengths) - 1
    length = max(lengths)
    res = []
    for i in range(len(all_continuation_ids)):
        c_ids = all_continuation_ids[i]
        local_res = []
        for j in range(length):
            t_idx = c_ids[j] if j < len(c_ids) else padding_value
            local_res.append(logits[i, start_idx + j, t_idx])
        res.append(local_res)
    return torch.tensor(res)


@torch.no_grad()
def score_batch_iterative(
        forward_func: Callable[[torch.Tensor], torch.Tensor],
        input_ids: torch.Tensor,
        all_continuation_ids: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    res = []
    for continuation_ids in all_continuation_ids:
        generator = _sample_score(
            forward_func=forward_func,
            input_ids=input_ids,
            continuation_ids=continuation_ids
        )
        all_logits = []
        for token, logits in generator:
            all_logits.append(logits[:, token.item()])
        res.append(torch.cat(all_logits))
    return res
