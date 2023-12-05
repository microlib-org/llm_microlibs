from typing import Callable, List

import torch
from tokenizers import Tokenizer
from tqdm import tqdm


def generate(
        input_texts: List[str],
        tokenizer: Tokenizer,
        forward_full_sequence_fn: Callable[[torch.Tensor], torch.Tensor],
        forward_single_fn: Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor],
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        max_new_tokens: int
) -> List[str]:
    pad_token = tokenizer.token_to_id('<|endoftext|>')
    all_input_ids = [torch.tensor(sample.ids) for sample in tokenizer.encode_batch(input_texts)]
    batch = torch.nn.utils.rnn.pad_sequence(all_input_ids, padding_value=pad_token, batch_first=True)
    input_ids = batch.cuda()
    n = torch.tensor([ids.shape for ids in all_input_ids])
    outputs = forward_full_sequence_fn(input_ids)
    next_tokens = sample_fn(outputs)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=(-1))
    for i in tqdm(range(max_new_tokens - 1)):
        outputs = forward_single_fn(input_ids[:, -1:], i, n)
        next_tokens = sample_fn(outputs)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=(-1))
    return tokenizer.decode_batch(input_ids.detach().cpu().numpy())
