import pytest
import transformers
from transformers import AutoTokenizer

from llm_falcon_model.configuration_RW import RWConfig
from llm_falcon_model.modelling_RW import RWForCausalLM


@pytest.fixture(scope='module')
def falcon_7b_pipeline():
    model = "tiiuae/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(model)

    config = RWConfig(
        vocab_size=65024,
        hidden_size=4544,
        n_layer=32,
        n_head=71
    )
    model = RWForCausalLM(config)
    return model


def test_forward():
    pass
