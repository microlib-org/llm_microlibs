import logging
from pathlib import Path
from typing import Union

from llm_sepweight.part_state_dict import PartStateDict, _load_verbose
from llm_sepweight.filenames import get_filenames


def load(path: Union[str, Path], spec: str) -> PartStateDict:
    path = Path(path)
    return PartStateDict.from_part_spec(path, spec)


def lazy_load(module: nn.Module, path: Union[str, Path], spec: str):
    path = Path(path)
    for child in get_filenames(spec):
        state_dict = _load_verbose(path, (path / spec).stem)
        module.load_state_dict(state_dict, strict=False)