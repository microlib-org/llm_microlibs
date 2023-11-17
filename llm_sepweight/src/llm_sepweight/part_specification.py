from dataclasses import dataclass
from typing import List


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
            num_layers = int(parts[1])
            return cls(begin=True, mid=[range(num_layers)], end=True, is_full=True)
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
