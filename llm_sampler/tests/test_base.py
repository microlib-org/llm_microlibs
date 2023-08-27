import pytest
import torch

import transformers
from tqdm import tqdm
from transformers import AutoTokenizer

from llm_sampler import sample, sample_multiple_choice


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


def huggingface_tokenize(pipeline, input_text):
    input_ids = pipeline.tokenizer(input_text, padding=False, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(torch.device("cuda"))["input_ids"]
    return input_ids


@pytest.mark.parametrize("input_text, expected_text", [
    ("The argument went over ", "3 days"),
    ("Magnus Carlsen had won the World ", "#1"),
    ("We are looking for a Junior ", "/ Mid")
])
def test_sample(input_text, expected_text, falcon_7b_pipeline):
    pipeline = falcon_7b_pipeline
    input_ids = huggingface_tokenize(pipeline, input_text)
    generator = sample(
        forward_func=lambda x: pipeline.model(input_ids=x).logits,
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


@pytest.mark.parametrize("input_text, choices, expected_class", [
    ("The sentiment of the sentence 'I loved it' is '", ("positive", "negative"), 0),
    ("The sentiment of the sentence 'I hated it' is '", ("positive", "negative"), 1),
    ("The seniority level of the job position 'Rockstar Java developer' is '", ("Internship", "Junior", "Mid", "Senior", "Lead"), 3),
    ("The seniority level of the job position 'L1 Software Developer' is '", ("Internship", "Junior", "Mid", "Senior", "Lead"), 2),
])
def test_sample_multiple_choice(input_text, choices, expected_class, falcon_7b_pipeline):
    pipeline = falcon_7b_pipeline
    generator = sample_multiple_choice(
        forward_func=lambda x: pipeline.model(input_ids=x).logits,
        input_ids=huggingface_tokenize(pipeline, input_text),
        all_continuation_ids=[huggingface_tokenize(pipeline, choice) for choice in choices]
    )
    raw_seqs = list(generator)
    results = torch.tensor([raw_seq[0] for raw_seq in raw_seqs]).float().softmax(dim=0).numpy()
    assert results.argmax() == expected_class

