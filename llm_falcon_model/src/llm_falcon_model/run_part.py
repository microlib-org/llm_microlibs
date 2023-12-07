import logging

import socket_rpc

import llm_falcon_model
import llm_sepweight
from llm_falcon_model.distributed import FalconNode


def serve_node(node):
    server = socket_rpc.RPCServer('localhost', 61001)
    server.add_fn(node.clear_cache)
    server.add_fn(node.prepare_for_full_sequence)
    server.add_fn(node.prepare_for_single_forward)
    server.add_fn(node.forward)
    server.add_fn(node.forward_full_sequence)
    server.add_fn(node.forward_single)
    server.serve()


def main():
    model_name = '7b'
    spec = '16-32 e'
    device = 'cuda:0'
    path = f'/home/hvrigazov/llm-sepweights/falcon-{model_name}/'
    next_node_port = None
    final_node_port = 61002
    logging.basicConfig(level=logging.DEBUG)

    part = llm_falcon_model.init_part(model_name, spec, device)
    part_state_dict = llm_sepweight.load(path, spec)
    part.load_state_dict(part_state_dict.to_dict())

    kwargs = {
        'part': part,
        'device': device
    }
    if next_node_port is not None:
        kwargs['next_node'] = socket_rpc.RPCClient('localhost', next_node_port)
    elif final_node_port is not None:
        kwargs['final_node'] = socket_rpc.RPCClient('localhost', final_node_port)
    node = FalconNode(**kwargs)
    serve_node(node)


if __name__ == '__main__':
    main()
