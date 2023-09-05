from pathlib import Path

from tokenizers import Tokenizer


def load_tokenizer():
    json_file = Path(__file__).parent / f'falcon7b_tokenizer.json'
    return Tokenizer.from_file(str(json_file))
