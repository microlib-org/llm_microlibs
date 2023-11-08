import torch

from llm_sepweight.loading import load_np_dir_as_state_dict


def test_load_np_dir():
    test_dir = '/home/hvrigazov/llm-sepweights/falcon-7b/mid/0'
    state_dict = load_np_dir_as_state_dict(test_dir)
    for k, v in state_dict.items():
        assert isinstance(v, torch.Tensor)
        assert v.dtype == torch.bfloat16
