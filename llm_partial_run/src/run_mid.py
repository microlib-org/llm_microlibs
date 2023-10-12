import argparse
import logging
import sys
import time
import traceback
from typing import Callable

import numpy as np
import torch
from socket_rpc import rpc, rpc_call

from llm_falcon_model import load_config, FalconMid
from llm_weights_mmap import load_separated_checkpoint


def initialize_falcon(device: str, model_name: str, start_layer: int, end_layer: int, separated_weights_path: str):
    torch.set_default_device(torch.device(device))
    torch.set_default_dtype(torch.bfloat16)
    config = load_config(model_name)
    module = FalconMid(config, start_layer, end_layer)
    module.eval()
    with torch.device('cpu'):
        load_separated_checkpoint(
            model=module,
            ckpt_path=separated_weights_path,
            prefix='transformer.',
            raw_key='lm_head'
        )
    return module


def run_partial(
        initialization_func: Callable,
        model_name: str,
        device: str,
        start_layer: int,
        end_layer: int,
        separated_weights_path: str,
        host: str,
        port: int,
        next_host: str,
        next_port: int,
):
    module = initialization_func(device, model_name, start_layer, end_layer, separated_weights_path)

    next_layer = rpc_call(host=next_host, port=next_port)

    @rpc(host=host, port=port)
    def falcon_forward(computation_id: float, x: np.ndarray):
        with torch.inference_mode():
            start_t = time.time()
            logging.info(f'Received shape {x.shape}')
            x = torch.as_tensor(x, dtype=torch.bfloat16)
            x = module(x)
            logging.info(
                f' {time.time()} Took {time.time() - start_t}. Shape after forward: {x.shape}. Sending to next layer ...')
            try:
                next_layer(computation_id, x.detach().cpu().half().numpy())
                logging.info(f'Done processing.')
            except Exception as e:
                logging.error(traceback.format_exc())


def main(initialization_func):
    parser = argparse.ArgumentParser(description="Run part of a LLM")
    parser.add_argument("--model_name", type=str, help="Model name", required=True)
    parser.add_argument("--start_layer", type=int, help="Start layer", required=True)
    parser.add_argument("--end_layer", type=int, help="End layer", required=True)
    parser.add_argument("--separated_weights_path", type=str, help="Path to separated weights", required=True)
    parser.add_argument("--host", type=str, help="Host", required=True)
    parser.add_argument("--port", type=int, help="Port", required=True)
    parser.add_argument("--next_host", type=str, help="Next host", required=True)
    parser.add_argument("--next_port", type=int, help="Next port", required=True)
    parser.add_argument("--device", type=str, help="Device", default='cuda:0')

    args = parser.parse_args()
    run_partial(
        initialization_func=initialization_func,
        model_name=args.model_name,
        device=args.device,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        separated_weights_path=args.separated_weights_path,
        host=args.host,
        port=args.port,
        next_host=args.next_host,
        next_port=args.next_port
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)  # This will log all messages of level DEBUG and above.
    main(initialize_falcon)

