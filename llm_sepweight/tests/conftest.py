import os
from pathlib import Path

import pytest


@pytest.fixture
def sepweight_dir():
    value = os.getenv('PATH_TO_SEPWEIGHT_DIR')
    if value is None:
        raise ValueError('Environment variable "PATH_TO_SEPWEIGHT_DIR" must be test in your .test.env file.')
    return Path(value)
