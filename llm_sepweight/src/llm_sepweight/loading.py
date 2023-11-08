from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch


def load_np_dir_as_state_dict(path: Union[str, Path], dtype: torch.dtype):
    path = Path(path)
    res = {}
    for child in path.glob('*.npy'):
        res[child.stem] = torch.tensor(np.load(str(child)), dtype=dtype)
    return res


def load_mid_as_state_dict(
        path: Path,
        part: Tuple[str]
):
    start = int(part[1])
    end = int(part[2])
    for i in range(start, end):
        path / 'mid' / str(i)


def load_as_state_dict(
        path: Union[str, Path],
        part: Tuple[str]
):
    path = Path(path)
    assert len(part) >= 1, "Part must have at least one element, e.g. 'full', 'start', 'mid', or 'end'"
    assert part[0] in ['full', 'start', 'mid', 'end'], "Part must start with one of: 'full', 'start', 'mid' or 'end'"

    if part[0] == 'mid':
        return load_mid_as_state_dict(path, part)


