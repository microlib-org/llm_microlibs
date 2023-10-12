import argparse
import logging
import sys
import time
import traceback

import numpy as np
import torch
from socket_rpc import rpc

from llm_falcon_model import load_config, FalconEnd
from llm_weights_mmap import load_separated_checkpoint


def run_falcon_partial(
        model_name: str,
        device: str,
        start_layer: int,
        separated_weights_path: str,
        host: str,
        port: int,
):
    torch.set_default_device(torch.device(device))
    torch.set_default_dtype(torch.bfloat16)
    config = load_config(model_name)
    module = FalconEnd(config, start_layer)
    module.eval()
    with torch.device('cpu'):
        load_separated_checkpoint(
            model=module,
            ckpt_path=separated_weights_path,
            prefix='transformer.',
            raw_key='lm_head'
        )

    @rpc(host=host, port=port)
    def falcon_forward(computation_id: float, x: np.ndarray):
        with torch.inference_mode():
            start_t = time.time()
            logging.info(f'Received shape {x.shape}')
            x = torch.as_tensor(x, dtype=torch.bfloat16)
            x = module(x)
            logging.info(f'{time.time()} Took {time.time() - start_t}. Shape after forward: {x.shape}. Saving results ...')
            try:
                np.save(f'{computation_id}.npy', x.detach().cpu().half().numpy())
                logging.info(f'Done processing.')
            except Exception as e:
                logging.error(traceback.format_exc())


def main(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    run_falcon_partial(
        model_name=args.model_name,
        device=args.device,
        start_layer=args.start_layer,
        separated_weights_path=args.separated_weights_path,
        host=args.host,
        port=args.port,
    )


if __name__ == '__main__':
    # Creating parser
    parser = argparse.ArgumentParser(description="Run Falcon Partial")

    # Adding arguments
    parser.add_argument("--model_name", type=str, help="Model name", required=True)
    parser.add_argument("--start_layer", type=int, help="Start layer", required=True)
    parser.add_argument("--separated_weights_path", type=str, help="Path to separated weights", required=True)
    parser.add_argument("--host", type=str, help="Host", required=True)
    parser.add_argument("--port", type=int, help="Port", required=True)
    parser.add_argument("--device", type=str, help="Device", default='cuda:0')

    # Parsing arguments
    args = parser.parse_args()

    # Running main function with parsed arguments
    main(args)
