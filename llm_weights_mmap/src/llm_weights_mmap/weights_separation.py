import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description='Dump each key in a .pth file to a separate .npy file')
    parser.add_argument('--pth_files_dir', metavar='PTH_FILES_DIR', type=str, help='Path to the .pth files directory', required=True)
    parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str, help='Output directory to dump the keys', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pth_files_dir = Path(args.pth_files_dir)
    output_dir.mkdir(exist_ok=True)
    pth_files_dir.mkdir(exist_ok=True)

    model_name = pth_files_dir.name
    (output_dir / model_name).mkdir(exist_ok=True)
    for pth_file in sorted(pth_files_dir.glob('*.pth')):
        print(f'Processing pth file "{pth_file.resolve()}"')
        (output_dir / model_name / pth_file.stem).mkdir(exist_ok=True, parents=True)
        for k, v in torch.load(pth_file, map_location="cpu").items():
            print(f'\tProcessing key "{k}"')
            filename_path = output_dir / model_name / pth_file.stem / f'{k}.npy'
            output_arr = v.numpy() if v.dtype != torch.bfloat16 else v.half().numpy()
            np.save(filename_path, output_arr)


if __name__ == '__main__':
    main()
