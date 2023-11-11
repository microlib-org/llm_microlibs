import argparse
import logging
from llm_falcon_model import falcon_decider
from llm_sepweight.pth_conversion import convert_pth_files
from llm_sepweight.safetensors_conversion import convert_safetensors_files


def convert_small(in_path, out_path):
    convert_pth_files(in_path, out_path, falcon_decider, extension='bin')


def convert_180b(in_path, out_path):
    convert_safetensors_files(in_path, out_path, falcon_decider)


def main():
    # Set up logging with basic configuration
    logging.basicConfig(level=logging.INFO)

    # Create argument parser
    parser = argparse.ArgumentParser(description='Convert model files to a different format.')
    parser.add_argument('model_size', choices=['7b', '40b', '180b'],
                        help='The size of the model to convert (7b, 40b, or 180b).')
    parser.add_argument('in_path', help='The input path of the model files.')
    parser.add_argument('out_path', help='The output path for the converted model files.')

    # Parse arguments
    args = parser.parse_args()

    # Decide which conversion function to use based on the model size
    if args.model_size in ['7b', '40b']:
        logging.info(f"Converting a small model of size {args.model_size}.")
        convert_small(args.in_path, args.out_path)
    else:
        logging.info("Converting a large model of size 180b.")
        convert_180b(args.in_path, args.out_path)


if __name__ == '__main__':
    main()
