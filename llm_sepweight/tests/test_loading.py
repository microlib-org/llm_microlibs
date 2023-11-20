import llm_sepweight


def test_load(sepweight_dir):
    state_dict = llm_sepweight.load(sepweight_dir, 'b')
    assert len(state_dict) == 1
    assert 'word_embeddings.weight' in state_dict
