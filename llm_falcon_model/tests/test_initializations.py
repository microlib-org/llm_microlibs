import torch

from llm_falcon_model import FalconStart, load_config


def test_initialize_start():
    config = load_config('7b')
    torch.set_default_device(torch.device("cuda:0"))
    torch.set_default_dtype(torch.bfloat16)
    start = FalconStart(config, 4).eval()

