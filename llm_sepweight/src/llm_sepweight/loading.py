import logging
from pathlib import Path
from typing import Union, Tuple

import torch


def load_flat_dir_as_state_dict(path: Union[str, Path], prefix: str = ''):
    path = Path(path)
    assert path.is_dir(), f"The supplied directory {path} does not exist!"
    logging.info(f'Prefix: "{prefix}", loading pth directory: {path} ...')
    res = {}
    for child in path.glob('*.pth'):
        logging.info(f'Prefix: "{prefix}", loading pth key: {child} ...')
        v = torch.load(child, map_location='cpu')
        res[f'{prefix}{child.stem}'] = v
    return res


def load_mid_as_state_dict(
        path: Path,
        start: int,
        end: int,
        prefix: str = '',
):
    logging.info(f'Prefix: "{prefix}", loading layers {start} to {end} ...')
    state_dict = {}
    for i in range(start, end):
        layer_prefix = f'{prefix}h.{i}.'
        layer_state_dict = load_flat_dir_as_state_dict(path / str(i), layer_prefix)
        state_dict.update(layer_state_dict)
    return state_dict


def load_as_state_dict(
        path: Union[str, Path],
        part: Tuple
):
    path = Path(path)
    assert len(part) >= 1, "Part must have at least one element, e.g. 'full', 'start', 'mid', or 'end'"
    assert part[0] in ['full', 'start', 'mid', 'end'], "Part must start with one of: 'full', 'start', 'mid' or 'end'"

    if part[0] == 'mid':
        start = int(part[1])
        end = int(part[2])
        return load_mid_as_state_dict(path / 'mid', start, end)
    if part[0] == 'start':
        state_dict = load_flat_dir_as_state_dict(path / 'start')
        start = 0
        end = int(part[1])
        mid_state_dict = load_mid_as_state_dict(path / 'mid', start, end)
        return {**state_dict, **mid_state_dict}
    if part[0] == 'end':
        state_dict = load_flat_dir_as_state_dict(path / 'end')
        start = int(part[1])
        end = int(part[2])
        mid_state_dict = load_mid_as_state_dict(path / 'mid', start, end)
        return {**state_dict, **mid_state_dict}
    if part[0] == 'full':
        start_state_dict = load_flat_dir_as_state_dict(path / 'start', prefix='start.')
        start = 0
        end = int(part[1])
        mid_state_dict = load_mid_as_state_dict(path / 'mid', start, end, prefix='mid.')
        end_state_dict = load_flat_dir_as_state_dict(path / 'end', prefix='end.')
        return {**start_state_dict, **mid_state_dict, **end_state_dict}





