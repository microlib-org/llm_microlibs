import logging
from pathlib import Path

import safetensors
from tqdm import tqdm

from llm_sepweight import dump


def convert_safetensors_files(in_path, out_path, decider):
    in_path = Path(in_path)
    out_path = Path(out_path)
    for child in tqdm(in_path.glob('*.safetensors')):
        logging.info(f'Processing {child} ...')
        data = safetensors.safe_open(child, framework='pt')
        state_dict = {k: data.get_tensor(k).clone() for k in data.keys()}
        dump(
            decider=decider,
            out_path=out_path,
            state_dict=state_dict
        )
    logging.info(f'Done.')
