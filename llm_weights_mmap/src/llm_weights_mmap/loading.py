from time import time

import numpy as np
import torch
import logging


def find_cat_dim(shapes, target_shape):
    if len(target_shape) <= 1:
        if shapes[0] == target_shape:
            return None
        if target_shape[0] == shapes[0][0] and target_shape[0] == shapes[1][0]:
            return None
        return 0
    if shapes[:, 1].sum() == target_shape[1]:
        return 1
    return 0


def load_separated_checkpoint(model, ckpt_path):
    logging.info(f'Starting to load separated checkpoint from path "{ckpt_path}" ...')
    start_t = time()
    for k, v in model.state_dict().items():
        logging.info(f'Loading "{k}" from "{ckpt_path}"')
        arrays = []
        shapes = []
        for child_dir in sorted(list(ckpt_path.iterdir())):
            np_arr = np.load(child_dir / f'{k}.npy')
            shapes.append(np_arr.shape)
            arrays.append(torch.tensor(np_arr))
        target_shape = v.shape
        cat_dim = find_cat_dim(np.array(shapes), target_shape)
        weights = torch.cat(arrays, dim=cat_dim) if cat_dim is not None else arrays[0]
        model.load_state_dict({k: weights}, strict=False)
    logging.info(f'Done loading checkpoint. Took {time() - start_t}')
