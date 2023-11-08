import joblib
import torch

from llm_falcon_model import load_config, FalconFull
from llm_sepweight import load_as_state_dict
import state_dict_paths


def test_regression():
    config = load_config('7b')
    state_dict = load_as_state_dict(
        path=state_dict_paths.falcon_7b,
        part=('full', str(config.num_hidden_layers)),
        dtype=torch.bfloat16
    )
    torch.set_default_device(torch.device("cuda:0"))
    torch.set_default_dtype(torch.bfloat16)
    model = FalconFull(config).eval()
    model.load_state_dict(state_dict)
    reference = joblib.load(state_dict_paths.falcon_7b_ref)
    assert (model(reference['input_ids']) - reference['logits']).abs().mean().item() < 0.02
