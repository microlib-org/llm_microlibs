import pytest
import torch
import state_dict_paths

from tokenizers import Tokenizer

from llm_falcon_model import load_tokenizer
from llm_sampler import sample_multiple_choice

from llm_falcon_model.configuration_RW import load_config
from llm_falcon_model.modelling_RW import RWForCausalLM


@pytest.fixture(scope='module')
def model_7b():
    config = load_config('7b')
    model = RWForCausalLM(config).to(torch.bfloat16).cuda().eval()
    state_dict = torch.load(state_dict_paths.falcon_7b, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


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
def test_forward(model_7b, input_text, expected_class):
    tokenizer_7b = load_tokenizer()
    input_ids = huggingface_tokenize(tokenizer_7b, input_text)

    generator = sample_multiple_choice(
        forward_func=lambda x: model_7b(input_ids=x),
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
