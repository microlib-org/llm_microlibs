from typing import Optional

import torch
from torch import nn

from llm_sepweight import PartSpec


class LLMPart(nn.Module):

    def __init__(self, llm_module, part_spec: PartSpec):
        super().__init__()
        self.begin = begin
        self.mid = nn.ModuleDict({str(i): layer_class(config) for i in range(start_layer, end_layer)})
        self.end = end

    def forward(self, input_ids: Optional[torch.LongTensor]) -> torch.Tensor:
        x = self.begin(input_ids)
        for layer_idx, layer in self.mid.items():
            x = layer(x)
        x = self.end(x)
        return x
