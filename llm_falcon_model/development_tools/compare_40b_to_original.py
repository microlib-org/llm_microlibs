import logging

import torch
from tqdm import tqdm

import llm_falcon_model
import llm_sepweight
from llm_falcon_model.modeling_updated import prepare_for_forward_full_sequence


def load_model():
    part = llm_falcon_model.init_part('40b', 'f', 'cpu')
    part_specs = ['b', 'e']
    for i in range(len(part.mid)):
        part_specs.append(f'{i}-{i+1}')
    for part_spec in tqdm(part_specs):
        state_dict = llm_sepweight.load('/home/hvrigazov/llm-sepweights/falcon-40b', part_spec).to_dict()
        part.load_state_dict(state_dict, strict=False)
    return part


def main():
    logging.basicConfig(level=logging.DEBUG)
    reference = torch.load('reference_40b.pth')
    tokenizer = llm_falcon_model.load_tokenizer()
    model = load_model()
    input_ids = torch.tensor(tokenizer.encode("Magnus Carlsen won the World ").ids).unsqueeze(0).cpu()
    prepare_for_forward_full_sequence(input_ids, model.mid.values())
    with torch.inference_mode():
        logits_new = model(input_ids)
        assert torch.allclose(logits_new, reference['logits'].cpu())
        print('OK')


if __name__ == '__main__':
    main()
