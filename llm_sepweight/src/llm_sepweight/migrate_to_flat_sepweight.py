import argparse
import logging
import shutil
from os import PathLike, listdir
from pathlib import Path
from typing import Union, Tuple

import torch


def _load_flat_dir_as_state_dict(path: Union[str, Path], prefix: str = ''):
    path = Path(path)
    assert path.is_dir(), f"The supplied directory {path} does not exist!"
    logging.info(f'Prefix: "{prefix}", loading pth directory: {path} ...')
    res = {}
    for child in path.glob('*.pth'):
        logging.info(f'Prefix: "{prefix}", loading pth key: {child} ...')
        v = torch.load(child, map_location='cpu')
        res[f'{prefix}{child.stem}'] = v
    return res


def _load_mid_as_state_dict(
        path: Path,
        start: int,
        end: int,
        prefix: str = '',
):
    logging.info(f'Prefix: "{prefix}", loading layers {start} to {end} ...')
    state_dict = {}
    for i in range(start, end):
        layer_prefix = f'{prefix}h.{i}.'
        layer_state_dict = _load_flat_dir_as_state_dict(path / str(i), layer_prefix)
        state_dict.update(layer_state_dict)
    return state_dict


def _load_dir_as_state_dict(
        path: Union[str, Path],
        part: Tuple
):
    path = Path(path)
    assert len(part) >= 1, "Part must have at least one element, e.g. 'full', 'start', 'mid', or 'end'"
    assert part[0] in ['full', 'start', 'mid', 'end'], "Part must start with one of: 'full', 'start', 'mid' or 'end'"

    if part[0] == 'mid':
        start = int(part[1])
        end = int(part[2])
        return _load_mid_as_state_dict(path / 'mid', start, end)
    if part[0] == 'start':
        state_dict = _load_flat_dir_as_state_dict(path / 'start')
        start = 0
        end = int(part[1])
        mid_state_dict = _load_mid_as_state_dict(path / 'mid', start, end)
        return {**state_dict, **mid_state_dict}
    if part[0] == 'end':
        state_dict = _load_flat_dir_as_state_dict(path / 'end')
        start = int(part[1])
        end = int(part[2])
        mid_state_dict = _load_mid_as_state_dict(path / 'mid', start, end)
        return {**state_dict, **mid_state_dict}
    if part[0] == 'full':
        start_state_dict = _load_flat_dir_as_state_dict(path / 'start', prefix='start.')
        start = 0
        end = int(part[1])
        mid_state_dict = _load_mid_as_state_dict(path / 'mid', start, end, prefix='mid.')
        end_state_dict = _load_flat_dir_as_state_dict(path / 'end', prefix='end.')
        return {**start_state_dict, **mid_state_dict, **end_state_dict}


def migrate(path: Path):
    logging.info("Loading state dict of begin ...")
    state_dict = _load_dir_as_state_dict(path, ('start', 0))
    logging.info("Saving state dict of begin ...")
    torch.save(state_dict, path / 'begin.pth')
    logging.info("Cleaning up directory of begin ...")
    shutil.rmtree(path / 'start')
    n_layers = len(listdir(path / 'mid'))
    for child in (path / 'mid').iterdir():
        s = int(child.name)
        e = s + 1
        logging.info(f"Loading state dict of mid {s} ...")
        state_dict = _load_dir_as_state_dict(path, ('mid', s, e))
        logging.info(f"Saving state dict of mid {s} ...")
        torch.save(state_dict, path / f'mid.{str(s).zfill(5)}.pth')
        logging.info(f"Cleaning up directory of mid {s} ...")
        shutil.rmtree(child)
    shutil.rmtree(path / 'mid')
    logging.info("Loading state dict of end ...")
    state_dict = _load_dir_as_state_dict(path, ('end', n_layers, n_layers))
    logging.info("Saving state dict of end ...")
    torch.save(state_dict, path / 'end.pth')
    logging.info(f"Cleaning up directory of end ...")
    shutil.rmtree(path / 'end')


def main():
    parser = argparse.ArgumentParser(description="Script for migrating to flat sepweight.")
    parser.add_argument('path', type=str, help='Path to the directory containing the state dictionaries')
    args = parser.parse_args()
    path = Path(args.path)
    migrate(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
