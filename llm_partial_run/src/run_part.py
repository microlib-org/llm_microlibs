import argparse
import importlib
import logging
import socket
import time
import traceback
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict

import numpy as np
import torch
from socket_rpc import rpc, rpc_call


def parse_ranges(input_string: str) -> List[Tuple]:
    tokens = input_string.split()
    try:
        numbers = [int(token) for token in tokens]
    except ValueError:
        raise ValueError("Input string contains non-integer values. "
                         "You need to provide a list of intervals, e.g. '0 5 10 15'")
    if len(numbers) % 2 != 0:
        raise ValueError("Input string contains an odd number of integers. "
                         "You need to provide a list of intervals, e.g. '0 5 10 15'")
    ranges = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]
    if len(ranges) <= 0:
        raise ValueError("At least one interval is required.")
    if ranges[0][0] == 0 and len(ranges) > 1:
        raise ValueError("If the first interval starts with 0, only one interval is allowed.")
    range_sizes = [b - a for a, b in ranges]
    if len(set(range_sizes)) > 1:
        raise ValueError("Intervals are of different sizes. "
                         "You need to provide a list of intervals, e.g. '0 5 10 15'")
    return ranges


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
                    new_state_dict[k.replace(f'h.{i}', f'h.{new_i}')] = v
        res.append(new_state_dict)
    return res


class PartialRun:

    def __init__(
            self,
            module,
            layer_prefix,
            next_layer,
            weight_reload_tuple,
            out_path
    ):
        self.module = module
        self.layer_prefix = layer_prefix
        self.next_layer = next_layer
        if weight_reload_tuple is not None:
            self.next_state_dict, self.cpu_state_dicts = weight_reload_tuple
        else:
            self.next_state_dict = None
            self.cpu_state_dicts = None
        self.out_path = out_path

    @torch.inference_mode()
    def module_forward(self, computation_id: float, x: np.ndarray):
        start_t = time.time()
        logging.info(f'Received shape {x.shape}')
        dtype = torch.long if x.dtype == np.int64 else torch.bfloat16
        x = torch.as_tensor(x, dtype=dtype)
        x = self.module(x)
        logging.info(f' {time.time()} Took {time.time() - start_t}. Shape after forward: {x.shape}.')

        try:
            if self.next_layer is not None:
                logging.info(f'{self.layer_prefix} sending to next layer ...')
                self.next_layer(computation_id, x.detach().cpu().half().numpy())
                logging.info(f'{self.layer_prefix} are done processing.')
            else:
                np.save(self.out_path / f'{computation_id}.npy', x.detach().cpu().half().numpy())
            if self.cpu_state_dicts is not None:
                logging.info('Moving next state dict to GPU ...')
                start_t = time.time()
                self.module.load_state_dict(self.cpu_state_dicts[self.next_state_dict])
                self.next_state_dict = (self.next_state_dict + 1) % len(self.cpu_state_dicts)
                logging.info(f'Done. Moving took {time.time() - start_t}')
        except Exception as e:
            logging.error(traceback.format_exc())


def run_partial(
        init_part: Callable,
        model_name: str,
        device: str,
        layers: str,
        separated_weights_path: str,
        host: str,
        port: int,
        next_host: Optional[str],
        next_port: Optional[int],
        out_path: Optional[Path],
        buffer_size: Optional[int] = 10 * 1024 * 1024,
        layers_zfill: int = 3
):
    ranges: List[Tuple] = parse_ranges(layers)
    start_layer, end_layer = ranges[0]
    weight_reload_mode = len(ranges) > 1
    if weight_reload_mode:
        logging.info(f"Multiple intervals specified: '{layers}'. Load all state dicts on CPU first ...")
        cpu_state_dicts = load_cpu_state_dicts(init_part, model_name, separated_weights_path, ranges)
        next_state_dict = 1
        weight_reload_tuple = next_state_dict, cpu_state_dicts
    else:
        weight_reload_tuple = None
    module = init_part(model_name, start_layer, end_layer, separated_weights_path, device)
    layer_range_str = f'{str(start_layer).zfill(layers_zfill)}-{str(end_layer).zfill(layers_zfill)}'
    layer_prefix = f'Layers {layer_range_str} on {socket.gethostname()}'
    logging.info(f'{layer_prefix} are ready.')

    next_layer = rpc_call(host=next_host, port=next_port) if next_host is not None and next_port is not None else None
    runner = PartialRun(module, layer_prefix, next_layer, weight_reload_tuple, out_path)
    rpc(host=host, port=port, buffer_size=buffer_size)(runner.module_forward)


def get_function_from_string(function_path):
    module_path, function_name = function_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def main():
    parser = argparse.ArgumentParser(description="Run part of a LLM")
    parser.add_argument("--init_part", type=str,
                        help="Path to initialization function. For example 'llm_falcon_model.init_part'", required=True)
    parser.add_argument("--model_name", type=str, help="Model name", required=True)
    parser.add_argument("--layers", type=str, help="Start layer", required=True)
    parser.add_argument("--separated_weights_path", type=str, help="Path to separated weights", required=True)
    parser.add_argument("--host", type=str, help="Host", required=True)
    parser.add_argument("--port", type=int, help="Port", required=True)
    parser.add_argument("--device", type=str, help="Device", default='cuda:0')
    parser.add_argument("--next_host", type=str, help="Next host", required=False)
    parser.add_argument("--next_port", type=int, help="Next port", required=False)
    parser.add_argument("--out", type=Path, help="Output path, if final module", default='.', required=False)

    args = parser.parse_args()
    initialization_func = get_function_from_string(args.init_part)
    run_partial(
        init_part=initialization_func,
        model_name=args.model_name,
        device=args.device,
        layers=args.layers,
        separated_weights_path=args.separated_weights_path,
        host=args.host,
        port=args.port,
        next_host=args.next_host,
        next_port=args.next_port,
        out_path=args.out
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)  # This will log all messages of level DEBUG and above.
    main()
