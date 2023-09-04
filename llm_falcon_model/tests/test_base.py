import pytest
import torch
from llm_sampler import sample_multiple_choice
from transformers import AutoTokenizer

from llm_falcon_model.configuration_RW import read_config_from_json
from llm_falcon_model.modelling_RW import RWForCausalLM


@pytest.fixture(scope='module')
def model_7b():
    config = read_config_from_json('7b')
    model = RWForCausalLM(config).to(torch.bfloat16).cuda()
    state_dict = torch.load('./llm_sampler/notebooks/state_dict.pth', map_location="cpu")
    model.load_state_dict(state_dict)
    return model


@pytest.fixture(scope='module')
def tokenizer_7b():
    return AutoTokenizer.from_pretrained("tiiuae/falcon-7b")


def huggingface_tokenize(tokenizer, input_text):
    input_ids = tokenizer(input_text, padding=False, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(torch.device("cuda"))["input_ids"]
    return input_ids


@pytest.mark.parametrize("input_text, expected_class", [
    ("The sentiment of the review 'The price was low' is '", 0),
    ("The sentiment of the review 'The quality was low' is '", 1),
    ("The sentiment of the review 'The food was pretty good, staff was friendly' is '", 0),
    ("The sentiment of the review 'The food was pretty good, but the prices were very high and the staff was impolite' is '", 1)
])
def test_forward(model_7b, tokenizer_7b, input_text, expected_class):
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
