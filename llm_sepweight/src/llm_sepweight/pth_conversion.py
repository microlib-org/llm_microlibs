import logging
from pathlib import Path

import torch

from llm_sepweight import dump


def convert_pth_files(in_path, out_path, decider, extension='.pth'):
    in_path = Path(in_path)
    out_path = Path(out_path)
    for child in in_path.glob(f'*.{extension}'):
        logging.info(f'Processing {child} ...')
        state_dict = torch.load(child)
        dump(
            decider=decider,
            state_dict=state_dict,
            out_path=out_path
        )
    logging.info(f'Done.')
