import llm_sepweight


def test_load(sepweight_dir):
    state_dict = llm_sepweight.load(sepweight_dir, 'b')
    assert len(state_dict.keys()) == 1
    assert 'begin.word_embeddings.weight' in state_dict


def test_load_full(sepweight_dir):
    state_dict = llm_sepweight.load(sepweight_dir, 'f')
    assert len(state_dict.keys()) == 190
    assert 'begin.word_embeddings.weight' in state_dict