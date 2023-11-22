import pytest
import torch

from llm_falcon_model import FalconBegin, load_config


@pytest.mark.parametrize("model_name", ["7b", "40b", "180b"])
def test_initialize_start(model_name):
    config = load_config(model_name)
    torch.set_default_device(torch.device("cuda:0"))
    torch.set_default_dtype(torch.bfloat16)
    start = FalconBegin(config, 4).eval()
