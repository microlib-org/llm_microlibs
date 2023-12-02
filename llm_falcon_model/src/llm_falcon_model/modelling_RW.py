# Ported from the original falcon-7b implementation

from typing import Optional, Tuple

import torch.utils.checkpoint
from torch import nn
from torch.nn import LayerNorm

from llm_falcon_model.modeling_updated import DecoderSingleLayerNorm, FalconDecoderLayer


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ret = input @ self.weight.T
        if self.bias is None:
            return ret
        else:
            return ret + self.bias


def get_layer_class(model_generation):
    return FalconDecoderLayer
    if model_generation == '7b':
        return DecoderSingleLayerNorm
    # if model_generation == '40b':
    #     return DecoderTwoLayerNorm
    # if model_generation == '180b':
    #     return DecoderTwoLayerNorm
    raise ValueError(f"Unknown model generation: '{model_generation}'. Available: '7b', '40b', '180b'")


class RWModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        model_generation = config._name_or_path.split('/')[1].split('-')[1].lower()
        layer_class = get_layer_class(model_generation)
        self.h = nn.ModuleList([layer_class(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

    def forward(self, input_ids: Optional[torch.LongTensor]) -> Tuple[torch.Tensor, ...]:
        inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = inputs_embeds
        for i, block in enumerate(self.h):
            outputs = block(hidden_states)
            hidden_states = outputs[0]
        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class RWForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transformer = RWModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Optional[torch.LongTensor]) -> Tuple[torch.Tensor]:
        transformer_outputs = self.transformer(input_ids)
        lm_logits = self.lm_head(transformer_outputs)
        return lm_logits


class FalconBegin(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids: Optional[torch.LongTensor]) -> torch.Tensor:
        return self.word_embeddings(input_ids)


class FalconEnd(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: Optional[torch.Tensor]) -> torch.Tensor:
        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits
