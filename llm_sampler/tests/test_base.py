import pytest
import torch

import transformers
from tqdm import tqdm
from transformers import AutoTokenizer

from llm_sampler import sample


@pytest.fixture(scope='module')
def falcon_7b_pipeline():
    model = "tiiuae/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(model)

    return transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # device_map="auto",
        device=torch.device("cuda")
    )


@pytest.mark.parametrize("input_text, expected_text", [
    ("The argument went over ", "3 days"),
    ("Magnus Carlsen had won the World ", "#1"),
    ("We are looking for a Junior ", "/ Mid")
])
def test_sample(input_text, expected_text, falcon_7b_pipeline):
    pipeline = falcon_7b_pipeline
    input_ids = pipeline.tokenizer(input_text, padding=False, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(torch.device("cuda"))["input_ids"]
    generator = sample(
        forward_func= lambda x: pipeline.model(input_ids=x).logits,
        input_ids=input_ids,
        max_new_tokens=2,
        temperature=0.001
    )
    result_tokens = []
    for token in tqdm(generator):
        int_token = token.cpu().item()
        result_tokens.append(int_token)
    decoded = pipeline.tokenizer.decode(result_tokens, skip_special_tokens=True)
    assert expected_text == decoded
