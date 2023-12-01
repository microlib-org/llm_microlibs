from typing import Tuple, Optional, Union, List, Sequence

import torch
from torch import nn, Tensor


def _convert_to_rw_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    batch_size, num_heads, kv_length, head_dim = past_key_value[0][0].shape
    batch_size_times_num_heads = batch_size * num_heads
    # [batch_size, num_heads, kv_length, head_dim] -> [batch_size * num_heads, kv_length, head_dim]
    return tuple(
        (
            layer_past[0].view(batch_size_times_num_heads, kv_length, head_dim),
            layer_past[1].view(batch_size_times_num_heads, kv_length, head_dim),
        )
        for layer_past in past_key_value
    )


class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length) that can be multiplied with attention scores

    Parameters:
        is_causal (`bool`):
            Whether the attention mask should be a uni-directional (causal) or bi-directional mask.

        sliding_window (`int`, *optional*):
            Optionally, the sliding window masks can be created if `sliding_window` is defined to a positive integer.
    """

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_causal_4d(
            self,
            batch_size: int,
            query_length: int,
            key_value_length: int,
            dtype: torch.dtype = torch.float32,
            device: Union[torch.device, "str"] = "cpu",
    ) -> torch.Tensor:
        """
        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
        bias to upper right hand triangular matrix (causal mask).
        """
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")

        # If shape is not cached, create a new causal mask and cache it
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )

        return causal_4d_mask

    def to_4d(
            self,
            attention_mask_2d: torch.Tensor,
            query_length: int,
            key_value_length: Optional[int] = None,
            dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        input_shape = (attention_mask_2d.shape[0], query_length)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.device
        )
        expanded_4d_mask = expanded_attn_mask if causal_4d_mask is None else expanded_attn_mask + causal_4d_mask

        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(
            input_ids_shape: torch.Size,
            dtype: torch.dtype,
            device: torch.device,
            past_key_values_length: int = 0,
            sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window + 1

            context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
            mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _prepare_4d_causal_attention_mask(
        attention_mask: Optional[torch.Tensor],
        input_shape: Union[torch.Size, Tuple, List],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length, dtype=inputs_embeds.dtype
        )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask


def _convert_head_mask_to_5d(head_mask, num_hidden_layers):
    """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
    assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    # head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
    return head_mask


def get_head_mask(
        head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
) -> Tensor:
    if head_mask is not None:
        head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers

    return head_mask


def _convert_cache_to_standard_format(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
    num_heads, ...]))
    """
    batch_size_times_num_heads, kv_length, head_dim = past_key_value[0][0].shape
    # [batch_size * self.num_heads, kv_length, head_dim] -> [batch_size, num_heads, kv_length, head_dim]
    # Note that don't want to use self.num_attention_heads because the number of heads may vary depending
    # on whether we use multi_query attention.
    num_heads = batch_size_times_num_heads // batch_size
    return tuple(
        (
            layer_past[0].view(batch_size, num_heads, kv_length, head_dim),
            layer_past[1].view(batch_size, num_heads, kv_length, head_dim),
        )
        for layer_past in past_key_value
    )


@torch.inference_mode()
def forward(
        input_ids,
        word_embeddings: nn.Embedding,
        mid: Sequence[nn.Module],
        ln_f: nn.LayerNorm,
        use_cache=True,
        past_key_values=None,
        head_mask=None,
        position_ids=None,
        attention_mask=None,
):
    num_hidden_layers = len(mid)
    presents = ()
    batch_size, seq_length = input_ids.shape
    if past_key_values is None:
        past_key_values = tuple([None] * num_hidden_layers)
    else:
        past_key_values = _convert_to_rw_cache(past_key_values)
    head_mask = get_head_mask(head_mask, num_hidden_layers)

    inputs_embeds = word_embeddings(input_ids)
    hidden_states = inputs_embeds
    past_key_values_length = 0
    if past_key_values[0] is not None:
        past_key_values_length = past_key_values[0][0].shape[1]  # 1 because RW-cache, not standard format

    if position_ids is None:
        device = input_ids.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )
    for i, (block, layer_past) in enumerate(zip(mid, past_key_values)):
        outputs = block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask[i],
            use_cache=use_cache,
            alibi=None
        )
        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)
    hidden_states = ln_f(hidden_states)
    if presents is not None:
        presents = _convert_cache_to_standard_format(presents, batch_size)
    return hidden_states, presents
