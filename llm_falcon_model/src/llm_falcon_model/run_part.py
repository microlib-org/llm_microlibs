import argparse
import logging

import socket_rpc

import llm_falcon_model
import llm_sepweight
from llm_falcon_model.distributed import FalconNode


def serve_node(node, server_port):
    server = socket_rpc.RPCServer('localhost', server_port)
    server.add_fn(node.clear_cache)
    server.add_fn(node.prepare_for_full_sequence)
    server.add_fn(node.prepare_for_single_forward)
    server.add_fn(node.forward)
    server.add_fn(node.forward_full_sequence)
    server.add_fn(node.forward_single)
    server.serve()


def main():
    parser = argparse.ArgumentParser(description="Set up and run a FalconNode server.")
    parser.add_argument('--model', '-m', required=True, help='Name of the model, e.g., 7b')
    parser.add_argument('--spec', '-s', required=True, help='Specification, e.g., 16-32 e')
    parser.add_argument('--device', '-d', required=True, help='Device to use, e.g., cuda:0')
    parser.add_argument('--weights_path', '-w', required=True, help='Path to the model weights')
    parser.add_argument('--port', '-p', type=int, required=True, help='Port for the server to listen on')
    parser.add_argument('--next_port', '-n', type=int, required=True, help='Port of the next node in the chain')
    parser.add_argument('--final', '-f', action='store_true', help='Whether the next node is final.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    part = llm_falcon_model.init_part(args.model, args.spec, args.device)
    part_state_dict = llm_sepweight.load(args.weights_path, args.spec)
    part.load_state_dict(part_state_dict.to_dict())

    kwargs = {
        'part': part,
        'device': args.device
    }
    client = socket_rpc.RPCClient('localhost', args.next_port)
    key = 'final_node' if args.final else 'next_node'
    logging.info(args)
    kwargs[key] = client
    node = FalconNode(**kwargs)
    serve_node(node, args.port)


if __name__ == '__main__':
    main()
