import argparse
import logging
from pathlib import Path

import torch

from tqdm import tqdm


def remove_leading_index(path):
    filenames = sorted(Path(path).glob('mid.*'))
    for filename in tqdm(filenames):
        layer_state_dict = torch.load(filename)
        migrated = {}
        for k, v in layer_state_dict.items():
            new_k = ('.'.join(k.split('.')[2:]))
            migrated[new_k] = v
        torch.save(migrated, filename)


def main():
    parser = argparse.ArgumentParser(description="Script for removing the leading index in transformer layers")
    parser.add_argument('path', type=str, help='Path to the directory containing the state dictionaries')
    args = parser.parse_args()
    path = Path(args.path)
    remove_leading_index(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
