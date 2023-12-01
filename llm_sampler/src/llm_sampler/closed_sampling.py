from typing import Sequence

import torch


def create_batch_with_continuations(
        input_ids: torch.Tensor,
        all_continuation_ids: Sequence[torch.Tensor],
        padding_value: int
):
    if len(input_ids.shape) == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids.squeeze(0)
    intermediary_continuations = []
    for c_ids in all_continuation_ids:
        c_ids = c_ids.to(input_ids.device)
        intermediary_continuations.append(torch.cat((input_ids, c_ids)))
    batch = torch.nn.utils.rnn.pad_sequence(intermediary_continuations, padding_value=padding_value, batch_first=True)
    batch = batch.to(input_ids.device)
    return batch


def get_scores_of_continuations(
        logits: torch.Tensor,
        all_continuation_ids: Sequence[torch.Tensor],
        padding_value: int
) -> torch.Tensor:
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
