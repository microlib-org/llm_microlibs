from pathlib import Path
from typing import Union

from llm_sepweight.part_state_dict import PartStateDict


def load(path: Union[str, Path], spec: str) -> PartStateDict:
    return PartStateDict.from_part_spec(path, spec)


