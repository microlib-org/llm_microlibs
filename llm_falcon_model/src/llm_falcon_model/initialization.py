import torch
import logging

from llm_falcon_model.configuration_RW import load_config
from llm_falcon_model.modelling_RW import FalconMid, FalconEnd, FalconStart
from llm_weights_mmap import load_separated_checkpoint


def init_part(device: str, model_name: str, start_layer: int, end_layer: int, separated_weights_path: str):
    torch.set_default_device(torch.device(device))
    torch.set_default_dtype(torch.bfloat16)
    logging.info(f"Loading config for model {model_name} ...")
    config = load_config(model_name)
    logging.info(f"Total number of layers for model {model_name}: {config.num_hidden_layers}")
    if end_layer > config.num_hidden_layers:
        raise ValueError(f"Number of hidden layers is {config.num_hidden_layers}, but end layer is {end_layer}")
    if start_layer == 0:
        module = FalconStart(config, end_layer)
    elif end_layer == config.num_hidden_layers:
        module = FalconEnd(config, start_layer)
    else:
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
