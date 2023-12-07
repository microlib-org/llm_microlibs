import logging

import torch
import socket_rpc

import llm_falcon_model
import llm_sepweight

from llm_falcon_model.distributed import FalconNode


def main():
    model_name = '40b'
    spec = '0-17'
    device = 'cuda:0'
    path = f'/home/hvrigazov/llm-sepweights/falcon-{model_name}/'
    next_node_port = 61001
    final_node_port = 61002
    logging.basicConfig(level=logging.DEBUG)

    part = llm_falcon_model.init_part(model_name, spec, device)
    part_state_dict = llm_sepweight.load(path, spec)
    part.load_state_dict(part_state_dict.to_dict())

    client = socket_rpc.RPCClient('localhost', 61001)
    node = FalconNode(part, device, client)

    server = socket_rpc.RPCServer('localhost', 61001, 1 * 1024 * 1024)
    server.add_fn(node.clear_cache)
    server.add_fn(node.prepare_for_full_sequence)
    server.add_fn(node.prepare_for_single_forward)
    server.add_fn(node.forward)
    server.add_fn(node.forward_full_sequence)
    server.add_fn(node.forward_single)
    server.serve()


if __name__ == '__main__':
    main()
