import logging

from pathlib import Path

import argparse
import numpy as np
import torch


def separate_weights(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    input_dir.mkdir(exist_ok=True)
    model_name = input_dir.name
    (output_dir / model_name).mkdir(exist_ok=True)
    pth_files = list(input_dir.glob('*.pth'))
    bin_files = list(input_dir.glob('*.bin'))
    files = pth_files if len(pth_files) > 0 else bin_files
    for pth_file in sorted(files):
        logging.info(f'Processing pth file "{pth_file.resolve()}"')
        (output_dir / model_name / pth_file.stem).mkdir(exist_ok=True, parents=True)
        for k, v in torch.load(pth_file, map_location="cpu").items():
            logging.info(f'\tProcessing key "{k}"')
            filename_path = output_dir / model_name / pth_file.stem / f'{k}.npy'
            output_arr = v.numpy() if v.dtype != torch.bfloat16 else v.half().numpy()
            np.save(filename_path, output_arr)


def main():
    parser = argparse.ArgumentParser(description='Dump each key in a .pth or .bin file to a separate .npy file')
    parser.add_argument('--input', metavar='input', type=str, help='Path to the .pth files directory',
                        required=True)
    parser.add_argument('--output', metavar='output', type=str, help='Output directory to dump the keys',
                        required=True)
    args = parser.parse_args()
    separate_weights(args.input, args.output)


if __name__ == '__main__':
    main()
