from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class PartStateDict:
    begin: Optional[Dict[str, torch.Tensor]]
    mid: Dict[int, Dict[str, torch.Tensor]]
    main_range: Optional[range]
    end: Optional[Dict[str, torch.Tensor]]
