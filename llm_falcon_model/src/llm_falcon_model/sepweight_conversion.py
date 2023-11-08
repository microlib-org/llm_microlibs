import torch

from llm_sepweight import dump_to_directory, falcon_decider


def convert_7b(in_path, out_dir):
    for child in in_path.glob('*.bin'):
        state_dict = torch.load(child)
        dump_to_directory(
            decider=falcon_decider,
            state_dict=state_dict,
            out_dir=out_dir
        )