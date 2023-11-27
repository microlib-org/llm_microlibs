import json
import logging
from pathlib import Path


class FalconConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, FalconConfig(value))
            else:
                setattr(self, key, value)


def _dict_to_object(d):
    return FalconConfig(d)


def _recursive_dict_to_object(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = _recursive_dict_to_object(value)
    return _dict_to_object(dictionary)


def load_config(model_name: str):
    logging.info(f"Loading config for model {model_name} ...")
    assert ".." not in model_name, "Invalid model name!"
    json_file = Path(__file__).parent / f'config_{model_name}.json'
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    config = _recursive_dict_to_object(json.loads(text))
    if not hasattr(config, 'num_hidden_layers'):
        config.num_hidden_layers = config.n_layer
    if not hasattr(config, 'num_attention_heads'):
        config.num_attention_heads = config.n_head
    if not hasattr(config, 'n_head_kv') and hasattr(config, 'num_kv_heads'):
        config.n_head_kv = config.num_kv_heads
    config.rotary = not config.alibi
    config.head_dim = config.hidden_size // config.num_attention_heads
    logging.info(f"Done loading config for model {model_name} ...")
    return config
