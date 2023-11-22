import logging
from os import PathLike
from pathlib import Path
from typing import Union, Tuple

import torch

from llm_sepweight.part_specification import PartSpec
from llm_sepweight.part_state_dict import PartStateDict


def load(path: Union[str, Path], part_spec: str):
    return load_part_spec(path, PartSpec.from_string(part_spec)).to_dict()


def load_part_spec(path: Union[str, Path], part_spec: PartSpec) -> PartStateDict:
    path = Path(path)
    if part_spec.is_full:
        logging.info(f'Loading full state dict at path {path} ...')
        begin = _load_verbose(path, 'begin')
        end = _load_verbose(path, 'end')
        mid = {}
        for mid_file in sorted(path.glob('mid.*.pth')):
            layer_idx = int(mid_file.stem.split('.')[1])
            mid[layer_idx] = _load_verbose(path, mid_file.stem)
        main_range = range(min(mid), max(mid) + 1)
        return PartStateDict(begin=begin, mid=mid, main_range=main_range, end=end)
    begin = _load_verbose(path, 'begin') if part_spec.begin else None
    mid = {}
    for layer_range in part_spec.mid:
        for layer_idx in layer_range:
            stem = f'mid.{str(layer_idx).zfill(5)}'
            mid[layer_idx] = _load_verbose(path, stem)
    main_range = part_spec.mid[0] if len(part_spec.mid) > 0 else None
    end = _load_verbose(path, 'end') if part_spec.end else None
    return PartStateDict(begin=begin, mid=mid, main_range=main_range, end=end)


def _load_verbose(path, stem):
    logging.info(f'Loading "{stem}" at path {path} ...')
    state_dict = torch.load(path / f'{stem}.pth')
    logging.info(f'Done loading "{stem} at path {path} ...')
    return state_dict
