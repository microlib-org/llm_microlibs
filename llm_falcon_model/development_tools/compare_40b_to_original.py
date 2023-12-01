import logging
from time import time

import torch

import llm_falcon_model
import llm_sepweight


def main():
    logging.basicConfig(level=logging.DEBUG)
    path = '/home/hvrigazov/llm-sepweights/falcon-40b'
    reference = torch.load('reference_40b.pth')
    tokenizer = llm_falcon_model.load_tokenizer()

    begin = llm_falcon_model.init_part('40b', 'b', 'cuda:0')
    begin.load_state_dict(llm_sepweight.load(path, 'b').to_dict())

    end = llm_falcon_model.init_part('40b', 'e', 'cuda:1')
    end.load_state_dict(llm_sepweight.load(path, 'e').to_dict())

    part0 = llm_falcon_model.init_part('40b', '0-15 30-45', 'cuda:0')
    part0_state_dict = llm_sepweight.load(path, '0-15 30-45')

    part1 = llm_falcon_model.init_part('40b', '15-30 45-60', 'cuda:1')
    part1_state_dict = llm_sepweight.load(path, '15-30 45-60')

    input_ids = torch.tensor(tokenizer.encode("Magnus Carlsen won the World ").ids).unsqueeze(0).cuda()
    start_t = time()
    with torch.inference_mode():
        x = begin(input_ids)
        print('Begin')
        x = part0(x)
        print('part0.0')
        part0.load_state_dict(part0_state_dict.to_dict(range(30, 45)))
        x = x.cuda(1)
        print('part1.0')
        x = part1(x)
        part1.load_state_dict(part1_state_dict.to_dict(range(45, 60)))
        x = x.cuda(0)
        print('part0.1')
        x = part0(x)
        x = x.cuda(1)
        print('part1.1')
        x = part1(x)
        print('end')
        x = end(x)
        x = x.cuda()
        print('Done')
        print(abs(x - reference['logits'].cuda()).mean())
    print(time() - start_t)


if __name__ == '__main__':
    main()
