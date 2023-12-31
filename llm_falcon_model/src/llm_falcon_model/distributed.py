import logging
from time import time
from typing import Optional

import torch
from socket_rpc import RPCClient

from llm_falcon_model.modeling_updated import _prepare_4d_causal_attention_mask
from llm_sepweight import Part


class FalconNode:
    def __init__(
            self,
            part: Part,
            device: str,
            next_node: Optional[RPCClient] = None,
            final_node: Optional[RPCClient] = None
    ):
        self.part = part
        self.device = device
        self.next_node = next_node
        self.final_node = final_node

    def clear_cache(self):
        if self.next_node is not None:
            self.next_node.clear_cache()
        for block in self.part.mid.values():
            block.self_attention.layer_past = None

    def prepare_for_full_sequence(self, input_shape: torch.Size):
        if self.next_node is not None:
            self.next_node.prepare_for_full_sequence(input_shape)
        mid = list(self.part.mid.values())
        position_ids = torch.arange(0, input_shape[1], dtype=torch.long).unsqueeze(0)
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=None,
            input_shape=input_shape,
            device=self.device,
            past_key_values_length=0
        )
        for i, block in enumerate(mid):
            block.self_attention.attention_mask = attention_mask
            block.self_attention.position_ids = position_ids

    def prepare_for_single_forward(self, input_shape: torch.Size, position_ids: torch.Tensor):
        if self.next_node is not None:
            self.next_node.prepare_for_single_forward(input_shape, position_ids)
        mid = list(self.part.mid.values())
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=None,
            input_shape=input_shape,
            device=self.device,
            past_key_values_length=mid[0].self_attention.layer_past[0].shape[1]
            # 1 because RW-cache, not standard format
        )
        for i, block in enumerate(mid):
            block.self_attention.attention_mask = attention_mask
            block.self_attention.position_ids = position_ids

    @torch.inference_mode()
    def forward(self, x:  torch.Tensor):
        x = x.to(self.device)
        start_t = time()
        x = self.part(x)
        logging.info(f'Took {time() - start_t}')
        if self.next_node is not None:
            self.next_node.forward(x.cpu())
        if self.final_node is not None:
            self.final_node.receive_result(x.cpu())
        return x

    def forward_full_sequence(self, input_ids: torch.Tensor):
        self.clear_cache()
        self.prepare_for_full_sequence(input_ids.shape)
        return self.forward(input_ids)

    def forward_single(self, input_ids: torch.Tensor, i: int, n: torch.Tensor):
        self.prepare_for_single_forward(input_ids.shape, i + n)
        return self.forward(input_ids)
