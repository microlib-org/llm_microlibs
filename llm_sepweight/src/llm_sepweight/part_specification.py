from dataclasses import dataclass
from typing import List


@dataclass
class PartSpec:
    begin: bool
    mid: List[range]
    end: bool

    @classmethod
    def from_string(cls, raw: str):
        parts = raw.split()
        begin = 'b' == parts[0]
        end = 'e' == parts[-1]
        ranges = []
        for part in parts:
            start, stop = map(int, part.split('-'))
            ranges.append(range(start, stop))
        return cls(begin=begin, mid=ranges, end=end)
