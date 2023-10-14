import logging
from typing import List, Dict


def load_cpu_state_dicts(init_part, model_name, separated_weights_path, ranges) -> List[Dict]:
    res = []
    for s, e in ranges:
        logging.info(f'Initializing {model_name} {s}-{e} state dict ...')
        module = init_part(model_name, s, e, separated_weights_path, 'cpu')
        state_dict = module.state_dict()
        new_state_dict = {}
        replacement_idx = dict(zip(range(s, e), range(*ranges[0])))
        for i in range(s, e):
            new_i = replacement_idx[i]
            for k, v in state_dict.items():
                if k.startswith(f'h.{i}'):
                    new_state_dict[k.replace(f'h.{i}', f'{new_i}')] = v
        res.append(new_state_dict)
    return res
