from typing import Callable, Optional

import torch
from socket_rpc import RPCClient

from llm_falcon_model.modeling_updated import _prepare_4d_causal_attention_mask
from llm_sepweight import Part


class FalconNode:
    def __init__(self, part: Part, device: str, client: Optional[RPCClient] = None):
        self.part = part
        self.device = device
        self.client = client

    def clear_cache(self):
        if self.client is not None:
            self.client.clear_cache()
        for block in self.part.mid.values():
            block.self_attention.layer_past = None

    def prepare_for_full_sequence(self, input_shape: torch.Size):
        if self.client is not None:
            self.client.prepare_for_full_sequence(input_shape)
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
        if self.client is not None:
            self.client.prepare_for_single_forward(input_shape, position_ids)
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
        x = self.part(x)
        if self.client is not None:
            self.client.forward(x)
        return x

    def forward_full_sequence(self, input_ids):
        self.clear_cache()
        self.prepare_for_full_sequence(input_ids.shape)
        return self.forward(input_ids)

    def forward_single(self, input_ids, i, n):
        self.prepare_for_single_forward(input_ids[:, -1:].shape, i + n)
        return self.forward(input_ids[:, -1:])
