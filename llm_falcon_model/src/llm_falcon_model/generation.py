from typing import Callable, List

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from llm_falcon_model.distributed import FalconNode


def generate(
        input_texts: List[str],
        tokenizer: Tokenizer,
        node: FalconNode,
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        max_new_tokens: int
) -> List[str]:
    pad_token = tokenizer.token_to_id('<|endoftext|>')
    all_input_ids = [torch.tensor(sample.ids) for sample in tokenizer.encode_batch(input_texts)]
    batch = torch.nn.utils.rnn.pad_sequence(all_input_ids, padding_value=pad_token, batch_first=True)
    input_ids = batch.cuda()
    n = torch.tensor([ids.shape for ids in all_input_ids])
    node.clear_cache()
    node.prepare_for_full_sequence(input_ids.shape)
    outputs = node.forward(input_ids)
    logits = outputs[:, -1, :]
    next_tokens = sample_fn(logits)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=(-1))
    for i in tqdm(range(max_new_tokens - 1)):
        node.prepare_for_single_forward(input_ids[:, -1:].shape, i + n)
        outputs = node.forward(input_ids[:, -1:])
        logits = outputs[:, -1, :]
        next_tokens = sample_fn(logits)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=(-1))
    return tokenizer.decode_batch(input_ids.detach().cpu().numpy())
