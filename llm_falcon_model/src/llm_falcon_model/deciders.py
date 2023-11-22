from typing import List, Tuple


def falcon_decider(key: str) -> List[str]:
    components = key.split('.')
    if key.startswith('transformer.word_embeddings'):
        return ["begin", '.'.join(components[1:])]
    if key.startswith('transformer.h'):
        return ["mid", components[2], '.'.join(components[3:])]
    if key.startswith('transformer.ln_f'):
        return ["end",  '.'.join(components[1:])]
    if key.startswith('lm_head'):
        return ["end",  '.'.join(components[0:])]
    raise ValueError(f"Could not handle key: '{key}")