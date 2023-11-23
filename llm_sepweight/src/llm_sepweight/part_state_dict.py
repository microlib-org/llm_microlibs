import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union, List

import torch


def _load_verbose(path, stem):
    logging.info(f'Loading "{stem}" at path {path} ...')
    state_dict = torch.load(path / f'{stem}.pth')
    logging.info(f'Done loading "{stem} at path {path} ...')
    return state_dict


@dataclass
class PartSpec:
    begin: bool
    mid: List[range]
    end: bool
    is_full: bool

    @classmethod
    def from_string(cls, raw: str):
        parts = raw.split()
        assert len(parts) > 0, f"You need to provide at least one part to load: '{raw}'"
        if parts[0] == 'f':
            # If user wants to load full model
            return cls(begin=True, mid=[], end=True, is_full=True)
        begin = 'b' == parts[0]
        end = 'e' == parts[-1]
        ranges = []
        for part in parts:
            if '-' not in part:
                continue
            start, stop = map(int, part.split('-'))
            ranges.append(range(start, stop))
        if begin and len(ranges) > 0:
            assert ranges[0].start == 0, 'When loading begin, first range must start with 0'
        return cls(begin=begin, mid=ranges, end=end, is_full=False)


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

    @classmethod
    def from_part_spec(cls, path: Union[str, Path], spec: str):
        part_spec = PartSpec.from_string(spec)
        path = Path(path)
        if part_spec.is_full:
            logging.info(f'Loading full state dict at path {path} ...')
            begin = _load_verbose(path, 'begin')
            end = _load_verbose(path, 'end')
            mid = {}
            for mid_file in sorted(path.glob('mid.*.pth')):
                layer_idx = int(mid_file.stem.split('.')[1])
                mid[layer_idx] = _load_verbose(path, mid_file.stem)
            main_range = range(min(mid), max(mid) + 1)
            return PartStateDict(begin=begin, mid=mid, main_range=main_range, end=end)
        begin = _load_verbose(path, 'begin') if part_spec.begin else None
        mid = {}
        for layer_range in part_spec.mid:
            for layer_idx in layer_range:
                stem = f'mid.{str(layer_idx).zfill(5)}'
                mid[layer_idx] = _load_verbose(path, stem)
        main_range = part_spec.mid[0] if len(part_spec.mid) > 0 else None
        end = _load_verbose(path, 'end') if part_spec.end else None
        return PartStateDict(begin=begin, mid=mid, main_range=main_range, end=end)
