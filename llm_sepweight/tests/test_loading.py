import llm_sepweight
from llm_sepweight import PartSpec


def test_load_part_spec(sepweight_dir):
    state_dict = llm_sepweight.load_part_spec(sepweight_dir, PartSpec.from_string('b'))
    assert len(state_dict.begin) == 1
    assert 'word_embeddings.weight' in state_dict.begin


def test_load(sepweight_dir):
    state_dict = llm_sepweight.load(sepweight_dir, 'b')
    assert len(state_dict) == 1
    assert 'word_embeddings.weight' in state_dict
