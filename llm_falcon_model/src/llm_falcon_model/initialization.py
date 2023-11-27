import logging

import torch

from llm_falcon_model.modelling_RW import FalconBegin, FalconEnd, DecoderSingleLayerNorm, DecoderTwoLayerNorm
from llm_falcon_model.configuration_RW import load_config
from llm_sepweight.part_state_dict import PartSpec

part_kwargs = {
    '7b': {
        'begin': FalconBegin,
        'mid': DecoderSingleLayerNorm,
        'end': FalconEnd
    },
    '40b': {
        'begin': FalconBegin,
        'mid': DecoderTwoLayerNorm,
        'end': FalconEnd
    },
    '180b': {
        'begin': FalconBegin,
        'mid': DecoderTwoLayerNorm,
        'end': FalconEnd
    }
}


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

