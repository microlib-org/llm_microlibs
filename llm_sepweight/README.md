# llm_sepweight

The `llm_sepweight` microlib is designed to manage the weights of large language models (LLMs) by organizing them into a directory format called `sepweight`.

`sepweight` essentially mirrors the state dict of the LLM into the filesystems, meaning that you will (roughly) have one 
file per key in the state dict of the LLM.

This format enables the distributed execution of LLMs by separating the model weights into distinct segments that can be individually managed and accessed as needed.

The only dependencies are `numpy` and `torch`.

## Installation

```bash
pip install llm_sepweight
```

## Quick Example

To convert an existing state dict into `sepweight`, you need to provide:

* `decider` is a function which will be called for each key in the state dict, and has to decide whether that key should 
be part of the `start`, `mid`, or `end` section. [Example](https://github.com/microlib-org/llm_microlibs/blob/7bf91edcd3d9d4cdbb40187ccbf6c7d0913a956a/llm_falcon_model/src/llm_falcon_model/deciders.py#L4)
* `state_dict` - is just your usual PyTorch state dict
* `out_path` is the directory, in which you want the result to be stored.

```python
from llm_sepweight import dump_to_directory

dump_to_directory(
    decider=decider,
    state_dict=state_dict,
    out_path=out_path
)
```

You could have multiple state dicts (for example coming from multiple files), it's ok to call `dump_to_directory` with 
each of them. The result will be combined state dict of all the state dicts provided for a given `out_path`.

## Goal format

`llm_sepweight` allows you to convert different formats to its own directory format, which is very simple.
Let's have a look at an example:

```bash
└── weights_root
    ├── end
    │   └── lm_head.pth
    ├── mid
    │   ├── 0
    │   │   ├── keys.pth
    │   │   ├── queries.pth
    │   │   └── values.pth
    │   ├── 1
    │   │   ├── keys.pth
    │   │   ├── queries.pth
    │   │   └── values.pth
    │   ├── 2
    │   │   ├── keys.pth
    │   │   ├── queries.pth
    │   │   └── values.pth
    │   └── 3
    │       ├── keys.pth
    │       ├── queries.pth
    │       └── values.pth
    └── start
        └── embeddings.pth

8 directories, 14 files

```

All the weights are stored in a directory in usual `.pth` files.

The root directory contains exactly three child directories: `start`, `mid` and `end`.
* The subdirectory `start` contains all the weights needed to compute the initial embeddings, prior to the transformer layers.
* The subdirectory `mid` contains numbered subdirectories corresponding to the weights of each layer.
* The subdirectory `end` contains the weights needed to compute the final prediction of the LM head.

This format is very simple and allows great flexibility. For example, a node running layers 0 to 3 will only need the 
`start`, `mid/0`, `mid/1`, `mid/2` subdirectories.


## Why do we need it?

There are all sorts of different formats for storing the weights of an LLM - `.pth` files, `safetensors`, `H5`,
`arrow`, `GGUF`, etc.  

Moreover, there is a lot of difference in the naming of the transformer layers, of the start embedding, and of the final head.
`llm_sepweight` aims to provide functions, through which you can convert different formats into a separated weights format.
