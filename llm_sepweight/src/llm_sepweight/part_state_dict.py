from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class PartStateDict:
    begin: Optional[Dict[str, torch.Tensor]]
    mid: Dict[int, Dict[str, torch.Tensor]]
    main_range: Optional[range]
    end: Optional[Dict[str, torch.Tensor]]

    def to_dict(self, target_range: Optional[range] = None):
        target_range = target_range if target_range is not None else self.main_range
        state_dict = {}
        if self.begin is not None:
            for k, v in self.begin.items():
                state_dict[f'begin.{k}'] = v
        if target_range is not None:
            for layer_idx, layer_idx_for_module in zip(target_range, self.main_range):
                layer_state_dict = self.mid[layer_idx]
                for k, v in layer_state_dict.items():
                    state_dict[f'mid.{layer_idx_for_module}.{k}'] = v
        if self.end is not None:
            for k, v in self.end.items():
                state_dict[f'end.{k}'] = v
        return state_dict
