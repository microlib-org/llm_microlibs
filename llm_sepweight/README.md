# llm_sepweight

The `llm_sepweight` microlib is designed to manage the weights of large language models (LLMs) by organizing them into a directory format called `sepweight`. 

This format enables the distributed execution of LLMs by separating the model weights into distinct segments that can be individually managed and accessed as needed.

The only dependencies are `numpy` and `torch`.

## Installation

```bash
pip install llm_sepweight
```

## Goal format

`llm_sepweight` allows you to convert different formats to its own directory format, which is very simple.
Let's have a look at an example:

```bash
└── weights_root
    ├── end
    │   └── lm_head.npy
    ├── mid
    │   ├── 0
    │   │   ├── keys.npy
    │   │   ├── queries.npy
    │   │   └── values.npy
    │   ├── 1
    │   │   ├── keys.npy
    │   │   ├── queries.npy
    │   │   └── values.npy
    │   ├── 2
    │   │   ├── keys.npy
    │   │   ├── queries.npy
    │   │   └── values.npy
    │   └── 3
    │       ├── keys.npy
    │       ├── queries.npy
    │       └── values.npy
    └── start
        └── embeddings.npy

8 directories, 14 files

```

All the weights are stored in a directory in usual `.npy` files.

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
