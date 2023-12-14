from pathlib import Path
from typing import Union

from torch import nn

from llm_sepweight.filenames import get_filenames
from llm_sepweight.part_state_dict import PartStateDict, _load_verbose


def load(path: Union[str, Path], spec: str) -> PartStateDict:
    path = Path(path)
    return PartStateDict.from_part_spec(path, spec)


def lazy_load(module: nn.Module, path: Union[str, Path], spec: str):
    path = Path(path)
    for child in get_filenames(spec):
        state_dict = _load_verbose(path, (path / child).stem)
        module.load_state_dict(state_dict, strict=False)
