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
        begin = torch.load(path / 'begin.pth')
