# llm_sepweight

Here you can find a full list of the things you can do with `llm_sepweight`.

To perform the conversion to `sepweight`, you must supply a `decider` function.
A `decider` is a function which will be called for each key in the state dict, and has to decide whether that key
should be part of the `start`, `mid`, or `end` section.

The `decider` function is model-specific.

The `decider` function accepts a string as an input (a key in the state dict) and should return a list,
which should start with either `start`, `mid` or `end` and the last element of the list will be filename under which
the weights will be saved.
If the key is part of the `mid` section, the second element should be the number of the layer as a string.
Here are some examples for the Falcon model:
```python
from llm_falcon_model import falcon_decider
falcon_decider('transformer.word_embeddings.weight') # => ["start", "word_embeddings.weight"]
falcon_decider('transformer.h.0.self_attention.query_key_value.weight') # => ["mid", "0", "self_attention.query_key_value.weight"]
falcon_decider('transformer.h.7.self_attention.query_key_value.bias') # => ["mid", "7", "self_attention.query_key_value.bias"]
falcon_decider('lm_head.weight') # => ["end", "lm_head.weight"]
```

## Contents

Dumping API

1. [Dump PyTorch state dict](#dump-pytorch-state-dict)
2. [Dump .pth or .bin files](#dump-pth-or-bin-files)
3. [Dump safetensors to a directory](#dump-safetensors)

Loading API

1. [Load state dict needed for a part of the LLM](#load-state-dict-needed-for-a-part-of-the-llm)

###  Dump PyTorch state dict

To convert a normal PyTorch state dict to `sepweight` format, use the `llm_sepweight.dump` function.

Apart from the `decider` function, you of course also need to supply the `state_dict` and the `out_path`:

```python
import llm_sepweight

llm_sepweight.dump(
    decider=decider,
    state_dict=state_dict,
    out_path=out_path
)
```


### Dump .pth or .bin files

To convert a directory which contains multiple `.pth` or `.bin` files to `sepweight`, use `convert_pth_files`.

Example:

```python
from llm_sepweight.pth_conversion import convert_pth_files

convert_pth_files(
    in_path="<PATH_TO_DIR_CONTAINING_PTH_FILES",
    out_path="<PATH_TO_OUTPUT_DIRECTORY>",
    decider=...,# decider function should be supplied here
    extension='pth' # this should be either "bin" or "pth"
)
```

### Dump safetensors

To convert a directory which contains multiple `.safetensors` files to `sepweight`, use `convert_safetensors_files`.

Example:

```python
from llm_sepweight.safetensors_conversion import convert_safetensors_files

convert_safetensors_files(
    in_path="<PATH_TO_DIR_CONTAINING_SAFETENSORS_FILES",
    out_path="<PATH_TO_OUTPUT_DIRECTORY>",
    decider=..., #decider function should be supplied here
)
```

### Load state dict needed for a part of the LLM

To load state dict needed for a part of the LLM just use the `llm_sepweight.load` function.

A part spec must be provided, which is just a string, in which you specify which `begin`, `mid` and `end` layers you 
need.

0. If the part spec is `f`, then the full state dict will be loaded.
1. If the part spec starts with `b`, then the `begin` state dict will be loaded.
2. If the part spec contains `5-7` for example, then the state dict for layers 5 to 7 will be loaded.
3. If the part spec ends with `e`, then the `end` state dict will be loaded.

For example, this will load the start embeddings and layers up to layer no.5:

```python
import llm_sepweight

llm_sepweight.load(
    path='<PATH_TO_SEPWEIGHT_DIRECTORY',
    part_spec='b 0-5'
)
```

This will load transformer layers 7 to 12:

```python
import llm_sepweight

llm_sepweight.load(
    path='<PATH_TO_SEPWEIGHT_DIRECTORY',
    part_spec="7-12"
)
```

This will load layers 75 to 80 and the end layers:

```python
import llm_sepweight

llm_sepweight.load(
    path='<PATH_TO_SEPWEIGHT_DIRECTORY',
    part_spec='75-80 e'
)
```

This will load all layers:

```python
import llm_sepweight

llm_sepweight.load(
    path='<PATH_TO_SEPWEIGHT_DIRECTORY',
    part_spec='f'
)
```
