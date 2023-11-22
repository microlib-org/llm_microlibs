import torch
from torch import nn


class Part(nn.Module):

    def __init__(self, begin=None, mid=None, end=None, mid_range=None):
        super().__init__()
        self.begin = begin() if begin is not None else None
        if mid_range is not None:
            assert isinstance(mid_range, range), "mid_range must be a range object."
        self.mid = nn.ModuleDict({str(i): mid() for i in mid_range}) if mid_range is not None else None
        self.end = end() if end is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.begin is not None:
            x = self.begin(x)
        if self.mid is not None:
            for layer_idx, layer in self.mid.items():
                x = layer(x)
        if self.end is not None:
            x = self.end(x)
        return x
