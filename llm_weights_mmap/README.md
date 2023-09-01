# llm_weights_mmap

## Installation

```bash
pip install llm_weights_mmap
```

## What is it

The script is designed to read `.pth` (PyTorch model state dictionary) files from a given directory and dump each key in the `.pth` file to a separate `.npy` (NumPy array) file in an output directory.


## Why use it:

1. **Selective Loading**: The script allows you to extract specific keys (i.e., layers or parameters) from a `.pth` file, so you can load only the portions of the model that you're interested in.

2. **Memory Efficiency**: The script enables you to load individual keys one at a time, offering a more memory-efficient way to inspect or manipulate model weights, especially when working with large models that might otherwise consume a significant amount of memory.


## Dependencies

- `numpy`
- `torch`

## Usage

Run the script from the command line and specify the `.pth` files directory and the output directory:

```bash
python -m llm_weights_mmap.weights_separation \
  --pth_files_dir /path/to/pth_files_dir \ 
  --output_dir /path/to/output_dir
```

### Example

Suppose you have a bunch of `.pth` files in a directory: 

```bash
➜ ls llama-2-13b
checklist.chk  consolidated.00.pth  consolidated.01.pth  params.json
```

Run the script this way:

```bash
python -m llm_weights_mmap.weights_separation \
  --pth_files_dir llama-2-13b \ 
  --output_dir llama-2-13b-separated
```

