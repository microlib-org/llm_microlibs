import logging
from os import PathLike
from pathlib import Path
from typing import Union, Tuple

import torch

from llm_sepweight.part_specification import PartSpec


def load(path: Union[str, Path], part_spec: str):
    return load_part_spec(path, PartSpec.from_string(part_spec))


def load_part_spec(path: Union[str, Path], part: PartSpec):
    if part.is_full:
        logging.info(f'Loading full state dict at path {path} ...')
        raise NotImplementedError()
    state_dict = {}
    if part.begin:
        state_dict.update(_load_verbose(path, 'begin'))
    for layer_range in part.mid:
        for layer_idx in layer_range:
            state_dict.update(_load_verbose(path, f'mid.{str(layer_idx).zfill(5)}'))
    if part.end:
        state_dict.update(_load_verbose(path, 'end'))
    return state_dict


def _load_verbose(path, stem):
    logging.info(f'Loading "{stem}" at path {path} ...')
    state_dict = torch.load(path / f'{stem}.pth')
    logging.info(f'Done loading "{stem} at path {path} ...')
    return state_dict
