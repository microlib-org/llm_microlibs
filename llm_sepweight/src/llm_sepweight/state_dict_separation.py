from pathlib import Path
from typing import Dict, Union, Callable

import numpy as np
import torch
import logging


def dump_to_directory(
        decider: Callable,
        state_dict: Dict[str, torch.Tensor],
        out_dir: Union[str, Path]
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f'Total number of keys: {len(state_dict)}')
    for k, v in state_dict.items():
        logging.info(f'Processing key "{k}" ...')
        goal_path = decider(k)
        logging.info(f'Goal path for "{k}" is {goal_path}.')
        assert len(goal_path) > 0, "Decider function should return a non-empty list."
        assert goal_path[0] in ['start', 'mid', 'end'], 'Goal path first element should be in ["start", "mid", "end"]'
        np_dir = out_dir / Path(*goal_path[:-1])
        np_dir.mkdir(exist_ok=True, parents=True)
        stemmed_filename = goal_path[-1]
        logging.info(f'Saving "{k}" as a numpy file ...')
        np.save(out_dir / f'{stemmed_filename}.npy', v.detach().cpu().half().numpy())
        logging.info(f'Done with "{k}".')


