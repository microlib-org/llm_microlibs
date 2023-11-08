import logging

from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch


def load_np_dir_as_state_dict(prefix: str, path: Union[str, Path], dtype: torch.dtype):
    path = Path(path)
    assert path.is_dir(), f"The supplied directory {path} does not exist!"
    logging.info(f'Prefix: "{prefix}", loading np directory: {path} ...')
    res = {}
    for child in path.glob('*.npy'):
        logging.info(f'Prefix: "{prefix}", loading np key: {child} ...')
        res[f'{prefix}.{child.stem}'] = torch.tensor(np.load(str(child)), dtype=dtype)
    return res


def load_mid_as_state_dict(
        prefix: str,
        path: Path,
        part: Tuple[str],
        dtype: torch.dtype
):
    start = int(part[1])
    end = int(part[2])
    logging.info(f'Prefix: "{prefix}", loading layers {start} to {end} ...')
    state_dict = {}
    for i in range(start, end):
        layer_prefix = f'{prefix}.h.{i}'
        layer_state_dict = load_np_dir_as_state_dict(layer_prefix, path / 'mid' / str(i), dtype)
        state_dict.update(layer_state_dict)
    return state_dict


def load_as_state_dict(
        path: Union[str, Path],
        part: Tuple[str],
        dtype: torch.dtype
):
    path = Path(path)
    assert len(part) >= 1, "Part must have at least one element, e.g. 'full', 'start', 'mid', or 'end'"
    assert part[0] in ['full', 'start', 'mid', 'end'], "Part must start with one of: 'full', 'start', 'mid' or 'end'"

    if part[0] == 'mid':
        return load_mid_as_state_dict('mid', path, part, dtype)

