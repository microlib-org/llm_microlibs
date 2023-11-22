# llm_sepweight

The `llm_sepweight` microlib is designed to manage the weights of large language models (LLMs) by organizing them into
a unified format called `sepweight`.

Every LLM has roughly the same three parts:
1. `begin` - the part of the model which computes the embeddings before the layers
2. `mid` - a number of (most commonly transformer) layers
3. `end` - the part of the model which converts the hidden state into a prediction for the next token

`sepweight` essentially mirrors the state dict of the LLM into the filesystems, meaning that you will (roughly) have one 
file per each of these components of the LLM.

This format enables the distributed execution of LLMs by separating the model weights into distinct segments that can
be individually managed and accessed as needed.

This microlib provides methods for:
1. `dump`-ing from different formats to `sepweight`
2. `load`-ing the state dict needed to run a part of the LLM

The only dependency is `torch`.

Microlib docs are available on [https://microlib.org/llm_sepweight.html](#https://microlib.org/llm_sepweight.html)

## Installation

```bash
pip install llm_sepweight
```

[![Downloads](https://static.pepy.tech/badge/llm_sepweight/month)](https://pepy.tech/project/llm_sepweight)
[![PyPi version](https://badgen.net/pypi/v/llm_sepweight/)](https://pypi.com/project/llm_sepweight)
[![PyPI license](https://img.shields.io/pypi/l/llm_sepweight.svg)](https://pypi.python.org/pypi/llm_sepweight/)

[![Read the Docs Badge](https://img.shields.io/badge/Read%20the%20Docs-8CA1AF?logo=readthedocs&logoColor=fff&style=for-the-badge)](https://microlib.org/llm_sepweight.html)

## Quick Example

To convert an existing state dict into `sepweight`, you need to provide:

* `decider` is a function which will be called for each key in the state dict, and has to decide whether that key should 
be part of the `begin`, `mid`, or `end` section and the the new name of the key.
[Example](https://github.com/microlib-org/llm_microlibs/blob/7bf91edcd3d9d4cdbb40187ccbf6c7d0913a956a/llm_falcon_model/src/llm_falcon_model/deciders.py#L4)
* `state_dict` - is just your usual PyTorch state dict
* `out_path` is the directory, in which you want the result to be stored.

```python
import llm_sepweight

llm_sepweight.dump(
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
├── begin.pth
├── end.pth
├── mid.00000.pth
├── mid.00001.pth
├── mid.00002.pth
├── mid.00003.pth
├── mid.00004.pth
└── mid.00005.pth

```

All the weights are stored in a directory in usual `.pth` files.

This format is very simple and allows great flexibility. For example, a node running layers 0 to 3 would only need to
download the `begin`, `mid.00000`,  `mid.00001`,  `mid.00000` files.


## Why do we need it?

There are all sorts of different formats for storing the weights of an LLM - `.pth` files, `safetensors`, `H5`,
`arrow`, `GGUF`, etc.  

Moreover, there is a lot of difference in the naming of the transformer layers, of the start embedding, and of the final head.
`llm_sepweight` aims to provide functions, through which you can convert different formats into a `sepweight` format.
The `sepweight` format is a unified, simple format that allows you to treat the weights of all LLMs in the same way
when running nodes in distributed way.
