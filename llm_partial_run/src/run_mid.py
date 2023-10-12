import argparse
import importlib
import logging
import socket
import time
import traceback
from typing import Callable

import numpy as np
import torch
from socket_rpc import rpc, rpc_call


def run_partial(
        initialization_func: Callable,
        model_name: str,
        device: str,
        layers: str,
        separated_weights_path: str,
        host: str,
        port: int,
        next_host: str,
        next_port: int,
        layers_zfill: int = 3
):
    layers_list = tuple(map(int, layers.split(' ')))
    start_layer, end_layer = layers_list
    module = initialization_func(device, model_name, start_layer, end_layer, separated_weights_path)
    layer_range_str = f'{str(start_layer).zfill(layers_zfill)}-{str(end_layer).zfill(layers_zfill)}'
    layer_prefix = f'Layers {layer_range_str} on {socket.gethostname()}'
    logging.info(f'{layer_prefix} are ready.')

    next_layer = rpc_call(host=next_host, port=next_port)

    @rpc(host=host, port=port)
    @torch.inference_mode()
    def module_forward(computation_id: float, x: np.ndarray):
        start_t = time.time()
        logging.info(f'Received shape {x.shape}')
        x = torch.as_tensor(x, dtype=torch.bfloat16)
        x = module(x)
        logging.info(f' {time.time()} Took {time.time() - start_t}. Shape after forward: {x.shape}.')
        try:
            logging.info(f'{layer_prefix} sending to next layer ...')
            next_layer(computation_id, x.detach().cpu().half().numpy())
            logging.info(f'{layer_prefix} are done processing.')
        except Exception as e:
            logging.error(traceback.format_exc())


def get_function_from_string(function_path):
    module_path, function_name = function_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def main():
    parser = argparse.ArgumentParser(description="Run part of a LLM")
    parser.add_argument("--init_fn", type=str, help="Path to initialization function. For example 'llm_falcon_model.initialize_part'", required=True)
    parser.add_argument("--model_name", type=str, help="Model name", required=True)
    parser.add_argument("--layers", type=int, help="Start layer", required=True)
    parser.add_argument("--separated_weights_path", type=str, help="Path to separated weights", required=True)
    parser.add_argument("--host", type=str, help="Host", required=True)
    parser.add_argument("--port", type=int, help="Port", required=True)
    parser.add_argument("--next_host", type=str, help="Next host", required=True)
    parser.add_argument("--next_port", type=int, help="Next port", required=True)
    parser.add_argument("--device", type=str, help="Device", default='cuda:0')

    args = parser.parse_args()
    initialization_func = get_function_from_string(args.init_fn)
    run_partial(
        initialization_func=initialization_func,
        model_name=args.model_name,
        device=args.device,
        layers=args.layers,
        separated_weights_path=args.separated_weights_path,
        host=args.host,
        port=args.port,
        next_host=args.next_host,
        next_port=args.next_port
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)  # This will log all messages of level DEBUG and above.
    main()

