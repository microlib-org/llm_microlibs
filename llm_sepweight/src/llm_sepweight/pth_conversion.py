import logging

import torch

from llm_sepweight import dump_to_directory


def convert_bin_files(in_path, out_path, decider):
    for child in in_path.glob('*.bin'):
        logging.info(f'Processing {child} ...')
        state_dict = torch.load(child)
        dump_to_directory(
            decider=decider,
            state_dict=state_dict,
            out_path=out_path
        )
    logging.info(f'Done.')
