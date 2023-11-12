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

1. [Dump PyTorch state dict to a directory](#dump-pytorch-state-dict-to-a-directory)
2. [Dump .pth or .bin files to a directory](#dump-pth-or-bin-files-to-a-directory)
3. [Dump safetensors to a directory](#dump-safetensors-to-a-directory)

Loading API

1. [Load state dict needed for a part of the LLM](#load-state-dict-needed-for-a-part-of-the-llm)

###  Dump PyTorch state dict to a directory

To convert a normal PyTorch state dict to `sepweight` format, use the `dump_to_directory` function.

Apart from the `decider` function, you of course also need to supply the `state_dict` and the `out_path`:

```python
from llm_sepweight import dump_to_directory

dump_to_directory(
    decider=decider,
    state_dict=state_dict,
    out_path=out_path
)
```


### Dump .pth or .bin files to a directory

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

### Dump safetensors to a directory

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

To load state dict needed for a part of the LLM just use the `load_as_state_dict` function.

For example, this will load the start embeddings and layers up to layer no.5:

```python
from llm_sepweight import load_as_state_dict

load_as_state_dict(
    path='<PATH_TO_SEPWEIGHT_DIRECTORY',
    part=("start", "5")
)
```

This will load transformer layers 7 to 12:

```python
from llm_sepweight import load_as_state_dict

load_as_state_dict(
    path='<PATH_TO_SEPWEIGHT_DIRECTORY',
    part=("mid", "7", "12")
)
```

This will load layers 75 to 80 and the end layers:

```python
from llm_sepweight import load_as_state_dict

load_as_state_dict(
    path='<PATH_TO_SEPWEIGHT_DIRECTORY',
    part=("end", "75", "80")
)
```
