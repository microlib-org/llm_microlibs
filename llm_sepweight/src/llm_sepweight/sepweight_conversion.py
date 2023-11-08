import safetensors
import torch
from tqdm import tqdm

from llm_sepweight import dump_to_directory


def convert_bin_files(in_path, out_path, decider):
    for child in in_path.glob('*.bin'):
        state_dict = torch.load(child)
        dump_to_directory(
            decider=decider,
            state_dict=state_dict,
            out_path=out_path
        )


def convert_safetensors_files(in_path, out_path, decider):
    for child in tqdm(in_path.glob('*.safetensors')):
        data = safetensors.safe_open(child, framework='pt')
        state_dict = {k: data.get_tensor(k).clone() for k in data.keys()}
        dump_to_directory(
            decider=decider,
            out_path=out_path,
            state_dict=state_dict
        )
