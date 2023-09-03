import json


class FalconConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, FalconConfig(value))
            else:
                setattr(self, key, value)


def dict_to_object(d):
    return FalconConfig(d)


def recursive_dict_to_object(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = recursive_dict_to_object(value)
    return dict_to_object(dictionary)


def read_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    config = recursive_dict_to_object(json.loads(text))
    config.num_hidden_layers = config.n_layer
    config.num_attention_heads = config.n_head
    config.rotary = not config.alibi
    config.head_dim = config.hidden_size // config.n_head
    return config
