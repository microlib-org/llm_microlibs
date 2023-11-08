import logging
from pathlib import Path
from typing import Dict, Union, Callable

import torch


def dump_to_directory(
        decider: Callable,
        state_dict: Dict[str, torch.Tensor],
        out_path: Union[str, Path]
):
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True, parents=True)
    logging.info(f'Total number of keys: {len(state_dict)}')
    for k, v in state_dict.items():
        logging.info(f'Processing key "{k}" ...')
        goal_path = decider(k)
        logging.info(f'Goal path for "{k}" is {goal_path}.')
        assert len(goal_path) > 0, "Decider function should return a non-empty list."
        assert goal_path[0] in ['start', 'mid', 'end'], 'Goal path first element should be in ["start", "mid", "end"]'
        np_dir = out_path / Path(*goal_path[:-1])
        np_dir.mkdir(exist_ok=True, parents=True)
        stemmed_filename = goal_path[-1]
        full_path = np_dir / f'{stemmed_filename}.pth'
        logging.info(f'Saving "{k}" as a pth file to {full_path} ...')
        torch.save(v.detach().cpu(), full_path)
        logging.info(f'Done with "{k}".')


