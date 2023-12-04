import torch

from llm_falcon_model.modeling_updated import _prepare_4d_causal_attention_mask
from llm_sepweight import Part


class FalconNode:
    def __init__(self, part: Part):
        self.part = part

    def clear_cache(self):
        for block in self.part.mid.values():
            block.self_attention.layer_past = None

    def prepare_for_full_sequence(self, input_shape: torch.Size, input_device: str):
        mid = list(self.part.mid.values())
        position_ids = torch.arange(0, input_shape[1], dtype=torch.long).unsqueeze(0)
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=None,
            input_shape=input_shape,
            device=input_device,
            past_key_values_length=0
        )
        for i, block in enumerate(mid):
            block.self_attention.attention_mask = attention_mask
            block.self_attention.position_ids = position_ids

    def prepare_for_single_forward(self, input_shape, input_device, token_idx):
        mid = list(self.part.mid.values())
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=None,
            input_shape=input_shape,
            device=input_device,
            past_key_values_length=mid[0].self_attention.layer_past[0].shape[1]
            # 1 because RW-cache, not standard format
        )
        for i, block in enumerate(mid):
            block.self_attention.attention_mask = attention_mask
            block.self_attention.position_ids = torch.tensor([[token_idx]])
