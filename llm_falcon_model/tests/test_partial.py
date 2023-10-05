from functools import partial
from pathlib import Path

import pytest
import torch
import state_dict_paths
from tokenizers import Tokenizer

from llm_falcon_model import load_tokenizer
from llm_falcon_model.configuration_RW import load_config
from llm_falcon_model.modelling_RW import FalconStart, FalconMid, FalconEnd
from llm_sampler import sample_multiple_choice
from llm_weights_mmap import load_separated_checkpoint


def compute_logits(input_ids, start, mid, end):
    x = start(input_ids)
    x = mid(x)
    x = end(x)
    return x


@pytest.fixture(scope='module')
def model_partial():
    path = Path(state_dict_paths.separated_falcon_7b)
    config = load_config('7b')
    start = FalconStart(config, 4).to(torch.bfloat16).cuda().eval()
    load_separated_checkpoint(start, path, prefix='transformer.')
    mid = FalconMid(config, 4, config.num_hidden_layers).to(torch.bfloat16).cuda().eval()
    load_separated_checkpoint(mid, path, prefix='transformer.')
    end = FalconEnd(config, config.num_hidden_layers).to(torch.bfloat16).cuda().eval()
    load_separated_checkpoint(end, path, prefix='transformer.', raw_key='lm_head')
    return partial(compute_logits, start=start, mid=mid, end=end)


def huggingface_tokenize(tokenizer: Tokenizer, input_text):
    input_ids = torch.tensor(tokenizer.encode(input_text, add_special_tokens=False).ids).unsqueeze(0)
    input_ids = input_ids.cuda()
    return input_ids


@pytest.mark.parametrize("input_text, expected_class", [
    ("The sentiment of the review 'The price was low' is '", 0),
    ("The sentiment of the review 'The quality was low' is '", 1),
    ("The sentiment of the review 'The food was pretty good, staff was friendly' is '", 0),
    ("The sentiment of the review 'The food was pretty good, but the prices were very high and the staff was impolite' is '", 1)
])
def test_forward(model_partial, input_text, expected_class):
    tokenizer_7b = load_tokenizer()
    input_ids = huggingface_tokenize(tokenizer_7b, input_text)

    generator = sample_multiple_choice(
        forward_func=model_partial,
        input_ids=input_ids,
        all_continuation_ids=[
            huggingface_tokenize(tokenizer_7b, "positive"),
            huggingface_tokenize(tokenizer_7b, "negative"),
            # huggingface_tokenize(tokenizer, "neutral")
        ]
    )
    predictions = torch.tensor([t[0] for t in generator]).softmax(dim=0)
    print(predictions.float().numpy())
    assert predictions.argmax().item() == expected_class
