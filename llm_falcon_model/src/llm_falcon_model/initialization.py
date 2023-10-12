import torch

from llm_falcon_model.configuration_RW import load_config
from llm_falcon_model.modelling_RW import FalconMid
from llm_weights_mmap import load_separated_checkpoint


def initialize_part(device: str, model_name: str, start_layer: int, end_layer: int, separated_weights_path: str):
    torch.set_default_device(torch.device(device))
    torch.set_default_dtype(torch.bfloat16)
    config = load_config(model_name)
    module = FalconMid(config, start_layer, end_layer)
    module.eval()
    with torch.device('cpu'):
        load_separated_checkpoint(
            model=module,
            ckpt_path=separated_weights_path,
            prefix='transformer.',
            raw_key='lm_head'
        )
    return module
