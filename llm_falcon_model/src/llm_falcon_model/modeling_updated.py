import math
from typing import Tuple, Optional, Union, List, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.falcon.modeling_falcon import FalconRotaryEmbedding, FalconLinearScalingRotaryEmbedding, \
    FalconDynamicNTKScalingRotaryEmbedding


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
        device,
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
            attention_mask, input_shape[-1], key_value_length, dtype=torch.bfloat16,
        )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=torch.bfloat16, device=device
        )

    return attention_mask


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


def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class FalconLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_states = input @ self.weight.T
        if self.bias is None:
            return hidden_states
        return hidden_states + self.bias

class FalconAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.is_causal = True

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.maybe_rotary = self._init_rope() if config.rotary else lambda q, k, t, p: (q, k)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        if config.new_decoder_architecture:
            qkv_out_dim = (config.num_kv_heads * 2 + config.num_attention_heads) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        self.query_key_value = FalconLinear(self.hidden_size, qkv_out_dim, bias=config.bias)
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        self.dense = FalconLinear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv_heads = config.num_kv_heads if (self.new_decoder_architecture or not self.multi_query) else 1

        self.layer_past = None

    def _init_rope(self):
        if self.config.rope_scaling is None:
            rotary_emb = FalconRotaryEmbedding(
                self.head_dim,
                base=self.config.rope_theta,
                max_position_embeddings=self.config.max_position_embeddings,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                rotary_emb = FalconLinearScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.config.rope_theta,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                rotary_emb = FalconDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.config.rope_theta,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
        return rotary_emb

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        if self.new_decoder_architecture:
            batch, seq_len, _ = fused_qkv.shape
            qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]
            key = torch.broadcast_to(key, query.shape)
            value = torch.broadcast_to(value, query.shape)

            query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
            return query, key, value
        elif not self.multi_query:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        else:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads + 2, self.head_dim)
            return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]

    # Copied from transformers.models.bloom.modeling_bloom.BloomAttention._merge_heads
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
    ):
        layer_past = self.layer_past
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(
            batch_size * num_kv_heads,
            query_length,
            self.head_dim,
        )
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_kv_heads, query_length, self.head_dim)

        past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
        query_layer, key_layer = self.maybe_rotary(query_layer, key_layer, past_kv_length, self.position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, kv_length, head_dim]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, kv_length, _ = key_layer.shape
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)

        if alibi is None:
            if hasattr(F, "scaled_dot_product_attention") and not output_attentions:
                attn_output = F.scaled_dot_product_attention(
                    query_layer_, key_layer_, value_layer_, self.attention_mask, 0.0, is_causal=False
                )
                attention_scores = None

            attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

            output_tensor = self.dense(attn_output)

            if output_attentions:
                return output_tensor, present, attention_scores
            else:
                self.layer_past = present
                return output_tensor, present


class FalconMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size

        self.dense_h_to_4h = FalconLinear(hidden_size, 4 * hidden_size, bias=config.bias)
        self.act = nn.GELU()
        self.dense_4h_to_h = FalconLinear(4 * hidden_size, hidden_size, bias=config.bias)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


class DecoderSingleLayerNorm(nn.Module):
    """
    Falcon-7B is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).

    The architecture is broadly adapted from the GPT-3 paper (Brown et al., 2020), with the following differences:

    Positionnal embeddings: rotary (Su et al., 2021);
    Attention: multiquery (Shazeer et al., 2019) and FlashAttention (Dao et al., 2022);
    Decoder-block: parallel attention/MLP with a single layer norm.
    """
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.self_attention = FalconAttention(config)
        self.mlp = FalconMLP(config)
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        self.input_layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

    def forward(
            self,
            hidden_states: torch.Tensor,
    ):
        residual = hidden_states
        attention_layernorm_out = self.input_layernorm(hidden_states)

        # Self attention.
        attn_outputs = self.self_attention(attention_layernorm_out)
        attention_output = attn_outputs[0]

        # MLP.
        mlp_output = self.mlp(attention_layernorm_out) + attention_output
        output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)
        return output


class FalconDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.self_attention = FalconAttention(config)
        self.mlp = FalconMLP(config)
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        if config.new_decoder_architecture:
            # The layer norm before self-attention
            self.ln_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            # The layer norm before the MLP
            self.ln_mlp = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        else:
            self.input_layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            if not config.parallel_attn:
                self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

    def forward(
            self,
            hidden_states: torch.Tensor,
            alibi: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            use_cache: bool = True,
            output_attentions: bool = False,
            **kwargs,
    ):
        residual = hidden_states

        if self.config.new_decoder_architecture:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        # Self attention.
        attn_outputs = self.self_attention(
            attention_layernorm_out,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        attention_output = attn_outputs[0]

        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    attention_output, residual, self.config.attention_dropout, training=self.training
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        outputs = attn_outputs[1:]

        # MLP.
        mlp_output = self.mlp(mlp_layernorm_out)

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output

        output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


def initialize_caches(seq_length):
    position_ids = torch.arange(0, seq_length, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0)
    return position_ids


@torch.inference_mode()
def forward_full_sequence(
        input_ids,
        word_embeddings: nn.Embedding,
        mid: Sequence[nn.Module],
        ln_f: nn.LayerNorm,
        lm_head: nn.Linear,
        attention_mask=None,
):
    position_ids = initialize_caches(input_ids.shape[1])
    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask=attention_mask,
        input_shape=input_ids.shape,
        device=input_ids.device,
        past_key_values_length=0
    )
    for i, block in enumerate(mid):
        block.self_attention.attention_mask = attention_mask
        block.self_attention.position_ids = position_ids
    hidden_states = word_embeddings(input_ids)
    for i, block in enumerate(mid):
        outputs = block(hidden_states)
        hidden_states = outputs[0]
    hidden_states = ln_f(hidden_states)
    lm_logits = lm_head(hidden_states)
    return lm_logits


@torch.inference_mode()
def forward(
        input_ids,
        word_embeddings: nn.Embedding,
        mid: Sequence[nn.Module],
        ln_f: nn.LayerNorm,
        lm_head: nn.Linear,
        position_ids,
):
    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask=None,
        input_shape=input_ids.shape,
        device=input_ids.device,
        past_key_values_length=mid[0].self_attention.layer_past[0].shape[1]  # 1 because RW-cache, not standard format
    )
    for i, block in enumerate(mid):
        block.self_attention.attention_mask = attention_mask
        block.self_attention.position_ids = position_ids
    hidden_states = word_embeddings(input_ids)
    for i, block in enumerate(mid):
        outputs = block(hidden_states)
        hidden_states = outputs[0]
    hidden_states = ln_f(hidden_states)
    lm_logits = lm_head(hidden_states)
    return lm_logits
