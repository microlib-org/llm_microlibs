import logging
from functools import partial
from typing import Callable

import torch

from llm_falcon_model.configuration_RW import load_config
from llm_falcon_model.modelling_RW import FalconBegin, FalconEnd, get_layer_class
from llm_sepweight.part_state_dict import PartSpec


def get_part_kwargs(model_name: str) -> dict[str, Callable]:
    config = load_config(model_name)
    res = {
        'begin': partial(FalconBegin, config=config),
        'mid': partial(get_layer_class(model_name), config=config),
        'end': partial(FalconEnd, config=config),
    }
    return res


def _create_part(
        model_name: str,
        spec: str,
        device: str
):
    assert model_name in ["7b", "40b", "180b"], 'Model name should be one of ["7b", "40b", "180b"]'
    with torch.device(device):
        logging.info(f"Loading config for model {model_name} ...")
        config = load_config(model_name)
        logging.info(f"Total number of layers for model {model_name}: {config.num_hidden_layers}")
        part_spec = PartSpec.from_string(spec)
        raise NotImplementedError()

