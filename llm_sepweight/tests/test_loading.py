import llm_sepweight


def test_load(sepweight_dir):
    state_dict = llm_sepweight.load(sepweight_dir, 'b').to_dict()
    assert len(state_dict.keys()) == 1
    assert 'begin.word_embeddings.weight' in state_dict


def test_load_full(sepweight_dir):
    state_dict = llm_sepweight.load(sepweight_dir, 'f').to_dict()
    assert len(state_dict.keys()) == 196
    assert 'begin.word_embeddings.weight' in state_dict