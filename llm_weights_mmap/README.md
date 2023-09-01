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
python -m llm_weights_mmap.weights_separation --pth_files_dir llama-2-13b --output_dir llama-2-13b-separated
```

Output: 

```bash
└── llama-2-13b
    ├── consolidated.00
    │   ├── layers.0.attention_norm.weight.npy
    │   ├── layers.0.attention.wk.weight.npy
    │   ├── layers.0.attention.wo.weight.npy
    │   ├── layers.0.attention.wq.weight.npy
    │   ├── layers.0.attention.wv.weight.npy
    │   ├── layers.0.feed_forward.w1.weight.npy
    │   ├── layers.0.feed_forward.w2.weight.npy
    │   ├── layers.0.feed_forward.w3.weight.npy
    │   ├── layers.0.ffn_norm.weight.npy
    │   ├── layers.10.attention_norm.weight.npy
    │   ├── layers.10.attention.wk.weight.npy
    │   ├── layers.10.attention.wo.weight.npy
    │   ├── layers.10.attention.wq.weight.npy
    │   ├── layers.10.attention.wv.weight.npy
    │   ├── layers.10.feed_forward.w1.weight.npy
    │   ├── layers.10.feed_forward.w2.weight.npy
    │   ├── layers.10.feed_forward.w3.weight.npy
    │   ├── layers.10.ffn_norm.weight.npy
    │   ├── layers.11.attention_norm.weight.npy
    │   ├── layers.11.attention.wk.weight.npy
    │   ├── layers.11.attention.wo.weight.npy
    │   ├── layers.11.attention.wq.weight.npy
    │   ├── layers.11.attention.wv.weight.npy
    │   ├── layers.11.feed_forward.w1.weight.npy
    │   ├── layers.11.feed_forward.w2.weight.npy
    │   ├── layers.11.feed_forward.w3.weight.npy
    │   ├── layers.11.ffn_norm.weight.npy
    │   ├── layers.12.attention_norm.weight.npy
    │   ├── layers.12.attention.wk.weight.npy
    │   ├── layers.12.attention.wo.weight.npy
    │   ├── layers.12.attention.wq.weight.npy
    │   ├── layers.12.attention.wv.weight.npy
    │   ├── layers.12.feed_forward.w1.weight.npy
    │   ├── layers.12.feed_forward.w2.weight.npy
    │   ├── layers.12.feed_forward.w3.weight.npy
    │   ├── layers.12.ffn_norm.weight.npy
    │   ├── layers.13.attention_norm.weight.npy
    │   ├── layers.13.attention.wk.weight.npy
    │   ├── layers.13.attention.wo.weight.npy
    │   ├── layers.13.attention.wq.weight.npy
    │   ├── layers.13.attention.wv.weight.npy
    │   ├── layers.13.feed_forward.w1.weight.npy
    │   ├── layers.13.feed_forward.w2.weight.npy
    │   ├── layers.13.feed_forward.w3.weight.npy
    │   ├── layers.13.ffn_norm.weight.npy
    │   ├── layers.14.attention_norm.weight.npy
    │   ├── layers.14.attention.wk.weight.npy
    │   ├── layers.14.attention.wo.weight.npy
    │   ├── layers.14.attention.wq.weight.npy
    │   ├── layers.14.attention.wv.weight.npy
    │   ├── layers.14.feed_forward.w1.weight.npy
    │   ├── layers.14.feed_forward.w2.weight.npy
    │   ├── layers.14.feed_forward.w3.weight.npy
    │   ├── layers.14.ffn_norm.weight.npy
    │   ├── layers.15.attention_norm.weight.npy
    │   ├── layers.15.attention.wk.weight.npy
    │   ├── layers.15.attention.wo.weight.npy
    │   ├── layers.15.attention.wq.weight.npy
    │   ├── layers.15.attention.wv.weight.npy
    │   ├── layers.15.feed_forward.w1.weight.npy
    │   ├── layers.15.feed_forward.w2.weight.npy
    │   ├── layers.15.feed_forward.w3.weight.npy
    │   ├── layers.15.ffn_norm.weight.npy
    │   ├── layers.16.attention_norm.weight.npy
    │   ├── layers.16.attention.wk.weight.npy
    │   ├── layers.16.attention.wo.weight.npy
    │   ├── layers.16.attention.wq.weight.npy
    │   ├── layers.16.attention.wv.weight.npy
    │   ├── layers.16.feed_forward.w1.weight.npy
    │   ├── layers.16.feed_forward.w2.weight.npy
    │   ├── layers.16.feed_forward.w3.weight.npy
    │   ├── layers.16.ffn_norm.weight.npy
    │   ├── layers.17.attention_norm.weight.npy
    │   ├── layers.17.attention.wk.weight.npy
    │   ├── layers.17.attention.wo.weight.npy
    │   ├── layers.17.attention.wq.weight.npy
    │   ├── layers.17.attention.wv.weight.npy
    │   ├── layers.17.feed_forward.w1.weight.npy
    │   ├── layers.17.feed_forward.w2.weight.npy
    │   ├── layers.17.feed_forward.w3.weight.npy
    │   ├── layers.17.ffn_norm.weight.npy
    │   ├── layers.18.attention_norm.weight.npy
    │   ├── layers.18.attention.wk.weight.npy
    │   ├── layers.18.attention.wo.weight.npy
    │   ├── layers.18.attention.wq.weight.npy
    │   ├── layers.18.attention.wv.weight.npy
    │   ├── layers.18.feed_forward.w1.weight.npy
    │   ├── layers.18.feed_forward.w2.weight.npy
    │   ├── layers.18.feed_forward.w3.weight.npy
    │   ├── layers.18.ffn_norm.weight.npy
    │   ├── layers.19.attention_norm.weight.npy
    │   ├── layers.19.attention.wk.weight.npy
    │   ├── layers.19.attention.wo.weight.npy
    │   ├── layers.19.attention.wq.weight.npy
    │   ├── layers.19.attention.wv.weight.npy
    │   ├── layers.19.feed_forward.w1.weight.npy
    │   ├── layers.19.feed_forward.w2.weight.npy
    │   ├── layers.19.feed_forward.w3.weight.npy
    │   ├── layers.19.ffn_norm.weight.npy
    │   ├── layers.1.attention_norm.weight.npy
    │   ├── layers.1.attention.wk.weight.npy
    │   ├── layers.1.attention.wo.weight.npy
    │   ├── layers.1.attention.wq.weight.npy
    │   ├── layers.1.attention.wv.weight.npy
    │   ├── layers.1.feed_forward.w1.weight.npy
    │   ├── layers.1.feed_forward.w2.weight.npy
    │   ├── layers.1.feed_forward.w3.weight.npy
    │   ├── layers.1.ffn_norm.weight.npy
    │   ├── layers.20.attention_norm.weight.npy
    │   ├── layers.20.attention.wk.weight.npy
    │   ├── layers.20.attention.wo.weight.npy
    │   ├── layers.20.attention.wq.weight.npy
    │   ├── layers.20.attention.wv.weight.npy
    │   ├── layers.20.feed_forward.w1.weight.npy
    │   ├── layers.20.feed_forward.w2.weight.npy
    │   ├── layers.20.feed_forward.w3.weight.npy
    │   ├── layers.20.ffn_norm.weight.npy
    │   ├── layers.21.attention_norm.weight.npy
    │   ├── layers.21.attention.wk.weight.npy
    │   ├── layers.21.attention.wo.weight.npy
    │   ├── layers.21.attention.wq.weight.npy
    │   ├── layers.21.attention.wv.weight.npy
    │   ├── layers.21.feed_forward.w1.weight.npy
    │   ├── layers.21.feed_forward.w2.weight.npy
    │   ├── layers.21.feed_forward.w3.weight.npy
    │   ├── layers.21.ffn_norm.weight.npy
    │   ├── layers.22.attention_norm.weight.npy
    │   ├── layers.22.attention.wk.weight.npy
    │   ├── layers.22.attention.wo.weight.npy
    │   ├── layers.22.attention.wq.weight.npy
    │   ├── layers.22.attention.wv.weight.npy
    │   ├── layers.22.feed_forward.w1.weight.npy
    │   ├── layers.22.feed_forward.w2.weight.npy
    │   ├── layers.22.feed_forward.w3.weight.npy
    │   ├── layers.22.ffn_norm.weight.npy
    │   ├── layers.23.attention_norm.weight.npy
    │   ├── layers.23.attention.wk.weight.npy
    │   ├── layers.23.attention.wo.weight.npy
    │   ├── layers.23.attention.wq.weight.npy
    │   ├── layers.23.attention.wv.weight.npy
    │   ├── layers.23.feed_forward.w1.weight.npy
    │   ├── layers.23.feed_forward.w2.weight.npy
    │   ├── layers.23.feed_forward.w3.weight.npy
    │   ├── layers.23.ffn_norm.weight.npy
    │   ├── layers.24.attention_norm.weight.npy
    │   ├── layers.24.attention.wk.weight.npy
    │   ├── layers.24.attention.wo.weight.npy
    │   ├── layers.24.attention.wq.weight.npy
    │   ├── layers.24.attention.wv.weight.npy
    │   ├── layers.24.feed_forward.w1.weight.npy
    │   ├── layers.24.feed_forward.w2.weight.npy
    │   ├── layers.24.feed_forward.w3.weight.npy
    │   ├── layers.24.ffn_norm.weight.npy
    │   ├── layers.25.attention_norm.weight.npy
    │   ├── layers.25.attention.wk.weight.npy
    │   ├── layers.25.attention.wo.weight.npy
    │   ├── layers.25.attention.wq.weight.npy
    │   ├── layers.25.attention.wv.weight.npy
    │   ├── layers.25.feed_forward.w1.weight.npy
    │   ├── layers.25.feed_forward.w2.weight.npy
    │   ├── layers.25.feed_forward.w3.weight.npy
    │   ├── layers.25.ffn_norm.weight.npy
    │   ├── layers.26.attention_norm.weight.npy
    │   ├── layers.26.attention.wk.weight.npy
    │   ├── layers.26.attention.wo.weight.npy
    │   ├── layers.26.attention.wq.weight.npy
    │   ├── layers.26.attention.wv.weight.npy
    │   ├── layers.26.feed_forward.w1.weight.npy
    │   ├── layers.26.feed_forward.w2.weight.npy
    │   ├── layers.26.feed_forward.w3.weight.npy
    │   ├── layers.26.ffn_norm.weight.npy
    │   ├── layers.27.attention_norm.weight.npy
    │   ├── layers.27.attention.wk.weight.npy
    │   ├── layers.27.attention.wo.weight.npy
    │   ├── layers.27.attention.wq.weight.npy
    │   ├── layers.27.attention.wv.weight.npy
    │   ├── layers.27.feed_forward.w1.weight.npy
    │   ├── layers.27.feed_forward.w2.weight.npy
    │   ├── layers.27.feed_forward.w3.weight.npy
    │   ├── layers.27.ffn_norm.weight.npy
    │   ├── layers.28.attention_norm.weight.npy
    │   ├── layers.28.attention.wk.weight.npy
    │   ├── layers.28.attention.wo.weight.npy
    │   ├── layers.28.attention.wq.weight.npy
    │   ├── layers.28.attention.wv.weight.npy
    │   ├── layers.28.feed_forward.w1.weight.npy
    │   ├── layers.28.feed_forward.w2.weight.npy
    │   ├── layers.28.feed_forward.w3.weight.npy
    │   ├── layers.28.ffn_norm.weight.npy
    │   ├── layers.29.attention_norm.weight.npy
    │   ├── layers.29.attention.wk.weight.npy
    │   ├── layers.29.attention.wo.weight.npy
    │   ├── layers.29.attention.wq.weight.npy
    │   ├── layers.29.attention.wv.weight.npy
    │   ├── layers.29.feed_forward.w1.weight.npy
    │   ├── layers.29.feed_forward.w2.weight.npy
    │   ├── layers.29.feed_forward.w3.weight.npy
    │   ├── layers.29.ffn_norm.weight.npy
    │   ├── layers.2.attention_norm.weight.npy
    │   ├── layers.2.attention.wk.weight.npy
    │   ├── layers.2.attention.wo.weight.npy
    │   ├── layers.2.attention.wq.weight.npy
    │   ├── layers.2.attention.wv.weight.npy
    │   ├── layers.2.feed_forward.w1.weight.npy
    │   ├── layers.2.feed_forward.w2.weight.npy
    │   ├── layers.2.feed_forward.w3.weight.npy
    │   ├── layers.2.ffn_norm.weight.npy
    │   ├── layers.30.attention_norm.weight.npy
    │   ├── layers.30.attention.wk.weight.npy
    │   ├── layers.30.attention.wo.weight.npy
    │   ├── layers.30.attention.wq.weight.npy
    │   ├── layers.30.attention.wv.weight.npy
    │   ├── layers.30.feed_forward.w1.weight.npy
    │   ├── layers.30.feed_forward.w2.weight.npy
    │   ├── layers.30.feed_forward.w3.weight.npy
    │   ├── layers.30.ffn_norm.weight.npy
    │   ├── layers.31.attention_norm.weight.npy
    │   ├── layers.31.attention.wk.weight.npy
    │   ├── layers.31.attention.wo.weight.npy
    │   ├── layers.31.attention.wq.weight.npy
    │   ├── layers.31.attention.wv.weight.npy
    │   ├── layers.31.feed_forward.w1.weight.npy
    │   ├── layers.31.feed_forward.w2.weight.npy
    │   ├── layers.31.feed_forward.w3.weight.npy
    │   ├── layers.31.ffn_norm.weight.npy
    │   ├── layers.32.attention_norm.weight.npy
    │   ├── layers.32.attention.wk.weight.npy
    │   ├── layers.32.attention.wo.weight.npy
    │   ├── layers.32.attention.wq.weight.npy
    │   ├── layers.32.attention.wv.weight.npy
    │   ├── layers.32.feed_forward.w1.weight.npy
    │   ├── layers.32.feed_forward.w2.weight.npy
    │   ├── layers.32.feed_forward.w3.weight.npy
    │   ├── layers.32.ffn_norm.weight.npy
    │   ├── layers.33.attention_norm.weight.npy
    │   ├── layers.33.attention.wk.weight.npy
    │   ├── layers.33.attention.wo.weight.npy
    │   ├── layers.33.attention.wq.weight.npy
    │   ├── layers.33.attention.wv.weight.npy
    │   ├── layers.33.feed_forward.w1.weight.npy
    │   ├── layers.33.feed_forward.w2.weight.npy
    │   ├── layers.33.feed_forward.w3.weight.npy
    │   ├── layers.33.ffn_norm.weight.npy
    │   ├── layers.34.attention_norm.weight.npy
    │   ├── layers.34.attention.wk.weight.npy
    │   ├── layers.34.attention.wo.weight.npy
    │   ├── layers.34.attention.wq.weight.npy
    │   ├── layers.34.attention.wv.weight.npy
    │   ├── layers.34.feed_forward.w1.weight.npy
    │   ├── layers.34.feed_forward.w2.weight.npy
    │   ├── layers.34.feed_forward.w3.weight.npy
    │   ├── layers.34.ffn_norm.weight.npy
    │   ├── layers.35.attention_norm.weight.npy
    │   ├── layers.35.attention.wk.weight.npy
    │   ├── layers.35.attention.wo.weight.npy
    │   ├── layers.35.attention.wq.weight.npy
    │   ├── layers.35.attention.wv.weight.npy
    │   ├── layers.35.feed_forward.w1.weight.npy
    │   ├── layers.35.feed_forward.w2.weight.npy
    │   ├── layers.35.feed_forward.w3.weight.npy
    │   ├── layers.35.ffn_norm.weight.npy
    │   ├── layers.36.attention_norm.weight.npy
    │   ├── layers.36.attention.wk.weight.npy
    │   ├── layers.36.attention.wo.weight.npy
    │   ├── layers.36.attention.wq.weight.npy
    │   ├── layers.36.attention.wv.weight.npy
    │   ├── layers.36.feed_forward.w1.weight.npy
    │   ├── layers.36.feed_forward.w2.weight.npy
    │   ├── layers.36.feed_forward.w3.weight.npy
    │   ├── layers.36.ffn_norm.weight.npy
    │   ├── layers.37.attention_norm.weight.npy
    │   ├── layers.37.attention.wk.weight.npy
    │   ├── layers.37.attention.wo.weight.npy
    │   ├── layers.37.attention.wq.weight.npy
    │   ├── layers.37.attention.wv.weight.npy
    │   ├── layers.37.feed_forward.w1.weight.npy
    │   ├── layers.37.feed_forward.w2.weight.npy
    │   ├── layers.37.feed_forward.w3.weight.npy
    │   ├── layers.37.ffn_norm.weight.npy
    │   ├── layers.38.attention_norm.weight.npy
    │   ├── layers.38.attention.wk.weight.npy
    │   ├── layers.38.attention.wo.weight.npy
    │   ├── layers.38.attention.wq.weight.npy
    │   ├── layers.38.attention.wv.weight.npy
    │   ├── layers.38.feed_forward.w1.weight.npy
    │   ├── layers.38.feed_forward.w2.weight.npy
    │   ├── layers.38.feed_forward.w3.weight.npy
    │   ├── layers.38.ffn_norm.weight.npy
    │   ├── layers.39.attention_norm.weight.npy
    │   ├── layers.39.attention.wk.weight.npy
    │   ├── layers.39.attention.wo.weight.npy
    │   ├── layers.39.attention.wq.weight.npy
    │   ├── layers.39.attention.wv.weight.npy
    │   ├── layers.39.feed_forward.w1.weight.npy
    │   ├── layers.39.feed_forward.w2.weight.npy
    │   ├── layers.39.feed_forward.w3.weight.npy
    │   ├── layers.39.ffn_norm.weight.npy
    │   ├── layers.3.attention_norm.weight.npy
    │   ├── layers.3.attention.wk.weight.npy
    │   ├── layers.3.attention.wo.weight.npy
    │   ├── layers.3.attention.wq.weight.npy
    │   ├── layers.3.attention.wv.weight.npy
    │   ├── layers.3.feed_forward.w1.weight.npy
    │   ├── layers.3.feed_forward.w2.weight.npy
    │   ├── layers.3.feed_forward.w3.weight.npy
    │   ├── layers.3.ffn_norm.weight.npy
    │   ├── layers.4.attention_norm.weight.npy
    │   ├── layers.4.attention.wk.weight.npy
    │   ├── layers.4.attention.wo.weight.npy
    │   ├── layers.4.attention.wq.weight.npy
    │   ├── layers.4.attention.wv.weight.npy
    │   ├── layers.4.feed_forward.w1.weight.npy
    │   ├── layers.4.feed_forward.w2.weight.npy
    │   ├── layers.4.feed_forward.w3.weight.npy
    │   ├── layers.4.ffn_norm.weight.npy
    │   ├── layers.5.attention_norm.weight.npy
    │   ├── layers.5.attention.wk.weight.npy
    │   ├── layers.5.attention.wo.weight.npy
    │   ├── layers.5.attention.wq.weight.npy
    │   ├── layers.5.attention.wv.weight.npy
    │   ├── layers.5.feed_forward.w1.weight.npy
    │   ├── layers.5.feed_forward.w2.weight.npy
    │   ├── layers.5.feed_forward.w3.weight.npy
    │   ├── layers.5.ffn_norm.weight.npy
    │   ├── layers.6.attention_norm.weight.npy
    │   ├── layers.6.attention.wk.weight.npy
    │   ├── layers.6.attention.wo.weight.npy
    │   ├── layers.6.attention.wq.weight.npy
    │   ├── layers.6.attention.wv.weight.npy
    │   ├── layers.6.feed_forward.w1.weight.npy
    │   ├── layers.6.feed_forward.w2.weight.npy
    │   ├── layers.6.feed_forward.w3.weight.npy
    │   ├── layers.6.ffn_norm.weight.npy
    │   ├── layers.7.attention_norm.weight.npy
    │   ├── layers.7.attention.wk.weight.npy
    │   ├── layers.7.attention.wo.weight.npy
    │   ├── layers.7.attention.wq.weight.npy
    │   ├── layers.7.attention.wv.weight.npy
    │   ├── layers.7.feed_forward.w1.weight.npy
    │   ├── layers.7.feed_forward.w2.weight.npy
    │   ├── layers.7.feed_forward.w3.weight.npy
    │   ├── layers.7.ffn_norm.weight.npy
    │   ├── layers.8.attention_norm.weight.npy
    │   ├── layers.8.attention.wk.weight.npy
    │   ├── layers.8.attention.wo.weight.npy
    │   ├── layers.8.attention.wq.weight.npy
    │   ├── layers.8.attention.wv.weight.npy
    │   ├── layers.8.feed_forward.w1.weight.npy
    │   ├── layers.8.feed_forward.w2.weight.npy
    │   ├── layers.8.feed_forward.w3.weight.npy
    │   ├── layers.8.ffn_norm.weight.npy
    │   ├── layers.9.attention_norm.weight.npy
    │   ├── layers.9.attention.wk.weight.npy
    │   ├── layers.9.attention.wo.weight.npy
    │   ├── layers.9.attention.wq.weight.npy
    │   ├── layers.9.attention.wv.weight.npy
    │   ├── layers.9.feed_forward.w1.weight.npy
    │   ├── layers.9.feed_forward.w2.weight.npy
    │   ├── layers.9.feed_forward.w3.weight.npy
    │   ├── layers.9.ffn_norm.weight.npy
    │   ├── norm.weight.npy
    │   ├── output.weight.npy
    │   ├── rope.freqs.npy
    │   └── tok_embeddings.weight.npy
    └── consolidated.01
        ├── layers.0.attention_norm.weight.npy
        ├── layers.0.attention.wk.weight.npy
        ├── layers.0.attention.wo.weight.npy
        ├── layers.0.attention.wq.weight.npy
        ├── layers.0.attention.wv.weight.npy
        ├── layers.0.feed_forward.w1.weight.npy
        ├── layers.0.feed_forward.w2.weight.npy
        ├── layers.0.feed_forward.w3.weight.npy
        ├── layers.0.ffn_norm.weight.npy
        ├── layers.10.attention_norm.weight.npy
        ├── layers.10.attention.wk.weight.npy
        ├── layers.10.attention.wo.weight.npy
        ├── layers.10.attention.wq.weight.npy
        ├── layers.10.attention.wv.weight.npy
        ├── layers.10.feed_forward.w1.weight.npy
        ├── layers.10.feed_forward.w2.weight.npy
        ├── layers.10.feed_forward.w3.weight.npy
        ├── layers.10.ffn_norm.weight.npy
        ├── layers.11.attention_norm.weight.npy
        ├── layers.11.attention.wk.weight.npy
        ├── layers.11.attention.wo.weight.npy
        ├── layers.11.attention.wq.weight.npy
        ├── layers.11.attention.wv.weight.npy
        ├── layers.11.feed_forward.w1.weight.npy
        ├── layers.11.feed_forward.w2.weight.npy
        ├── layers.11.feed_forward.w3.weight.npy
        ├── layers.11.ffn_norm.weight.npy
        ├── layers.12.attention_norm.weight.npy
        ├── layers.12.attention.wk.weight.npy
        ├── layers.12.attention.wo.weight.npy
        ├── layers.12.attention.wq.weight.npy
        ├── layers.12.attention.wv.weight.npy
        ├── layers.12.feed_forward.w1.weight.npy
        ├── layers.12.feed_forward.w2.weight.npy
        ├── layers.12.feed_forward.w3.weight.npy
        ├── layers.12.ffn_norm.weight.npy
        ├── layers.13.attention_norm.weight.npy
        ├── layers.13.attention.wk.weight.npy
        ├── layers.13.attention.wo.weight.npy
        ├── layers.13.attention.wq.weight.npy
        ├── layers.13.attention.wv.weight.npy
        ├── layers.13.feed_forward.w1.weight.npy
        ├── layers.13.feed_forward.w2.weight.npy
        ├── layers.13.feed_forward.w3.weight.npy
        ├── layers.13.ffn_norm.weight.npy
        ├── layers.14.attention_norm.weight.npy
        ├── layers.14.attention.wk.weight.npy
        ├── layers.14.attention.wo.weight.npy
        ├── layers.14.attention.wq.weight.npy
        ├── layers.14.attention.wv.weight.npy
        ├── layers.14.feed_forward.w1.weight.npy
        ├── layers.14.feed_forward.w2.weight.npy
        ├── layers.14.feed_forward.w3.weight.npy
        ├── layers.14.ffn_norm.weight.npy
        ├── layers.15.attention_norm.weight.npy
        ├── layers.15.attention.wk.weight.npy
        ├── layers.15.attention.wo.weight.npy
        ├── layers.15.attention.wq.weight.npy
        ├── layers.15.attention.wv.weight.npy
        ├── layers.15.feed_forward.w1.weight.npy
        ├── layers.15.feed_forward.w2.weight.npy
        ├── layers.15.feed_forward.w3.weight.npy
        ├── layers.15.ffn_norm.weight.npy
        ├── layers.16.attention_norm.weight.npy
        ├── layers.16.attention.wk.weight.npy
        ├── layers.16.attention.wo.weight.npy
        ├── layers.16.attention.wq.weight.npy
        ├── layers.16.attention.wv.weight.npy
        ├── layers.16.feed_forward.w1.weight.npy
        ├── layers.16.feed_forward.w2.weight.npy
        ├── layers.16.feed_forward.w3.weight.npy
        ├── layers.16.ffn_norm.weight.npy
        ├── layers.17.attention_norm.weight.npy
        ├── layers.17.attention.wk.weight.npy
        ├── layers.17.attention.wo.weight.npy
        ├── layers.17.attention.wq.weight.npy
        ├── layers.17.attention.wv.weight.npy
        ├── layers.17.feed_forward.w1.weight.npy
        ├── layers.17.feed_forward.w2.weight.npy
        ├── layers.17.feed_forward.w3.weight.npy
        ├── layers.17.ffn_norm.weight.npy
        ├── layers.18.attention_norm.weight.npy
        ├── layers.18.attention.wk.weight.npy
        ├── layers.18.attention.wo.weight.npy
        ├── layers.18.attention.wq.weight.npy
        ├── layers.18.attention.wv.weight.npy
        ├── layers.18.feed_forward.w1.weight.npy
        ├── layers.18.feed_forward.w2.weight.npy
        ├── layers.18.feed_forward.w3.weight.npy
        ├── layers.18.ffn_norm.weight.npy
        ├── layers.19.attention_norm.weight.npy
        ├── layers.19.attention.wk.weight.npy
        ├── layers.19.attention.wo.weight.npy
        ├── layers.19.attention.wq.weight.npy
        ├── layers.19.attention.wv.weight.npy
        ├── layers.19.feed_forward.w1.weight.npy
        ├── layers.19.feed_forward.w2.weight.npy
        ├── layers.19.feed_forward.w3.weight.npy
        ├── layers.19.ffn_norm.weight.npy
        ├── layers.1.attention_norm.weight.npy
        ├── layers.1.attention.wk.weight.npy
        ├── layers.1.attention.wo.weight.npy
        ├── layers.1.attention.wq.weight.npy
        ├── layers.1.attention.wv.weight.npy
        ├── layers.1.feed_forward.w1.weight.npy
        ├── layers.1.feed_forward.w2.weight.npy
        ├── layers.1.feed_forward.w3.weight.npy
        ├── layers.1.ffn_norm.weight.npy
        ├── layers.20.attention_norm.weight.npy
        ├── layers.20.attention.wk.weight.npy
        ├── layers.20.attention.wo.weight.npy
        ├── layers.20.attention.wq.weight.npy
        ├── layers.20.attention.wv.weight.npy
        ├── layers.20.feed_forward.w1.weight.npy
        ├── layers.20.feed_forward.w2.weight.npy
        ├── layers.20.feed_forward.w3.weight.npy
        ├── layers.20.ffn_norm.weight.npy
        ├── layers.21.attention_norm.weight.npy
        ├── layers.21.attention.wk.weight.npy
        ├── layers.21.attention.wo.weight.npy
        ├── layers.21.attention.wq.weight.npy
        ├── layers.21.attention.wv.weight.npy
        ├── layers.21.feed_forward.w1.weight.npy
        ├── layers.21.feed_forward.w2.weight.npy
        ├── layers.21.feed_forward.w3.weight.npy
        ├── layers.21.ffn_norm.weight.npy
        ├── layers.22.attention_norm.weight.npy
        ├── layers.22.attention.wk.weight.npy
        ├── layers.22.attention.wo.weight.npy
        ├── layers.22.attention.wq.weight.npy
        ├── layers.22.attention.wv.weight.npy
        ├── layers.22.feed_forward.w1.weight.npy
        ├── layers.22.feed_forward.w2.weight.npy
        ├── layers.22.feed_forward.w3.weight.npy
        ├── layers.22.ffn_norm.weight.npy
        ├── layers.23.attention_norm.weight.npy
        ├── layers.23.attention.wk.weight.npy
        ├── layers.23.attention.wo.weight.npy
        ├── layers.23.attention.wq.weight.npy
        ├── layers.23.attention.wv.weight.npy
        ├── layers.23.feed_forward.w1.weight.npy
        ├── layers.23.feed_forward.w2.weight.npy
        ├── layers.23.feed_forward.w3.weight.npy
        ├── layers.23.ffn_norm.weight.npy
        ├── layers.24.attention_norm.weight.npy
        ├── layers.24.attention.wk.weight.npy
        ├── layers.24.attention.wo.weight.npy
        ├── layers.24.attention.wq.weight.npy
        ├── layers.24.attention.wv.weight.npy
        ├── layers.24.feed_forward.w1.weight.npy
        ├── layers.24.feed_forward.w2.weight.npy
        ├── layers.24.feed_forward.w3.weight.npy
        ├── layers.24.ffn_norm.weight.npy
        ├── layers.25.attention_norm.weight.npy
        ├── layers.25.attention.wk.weight.npy
        ├── layers.25.attention.wo.weight.npy
        ├── layers.25.attention.wq.weight.npy
        ├── layers.25.attention.wv.weight.npy
        ├── layers.25.feed_forward.w1.weight.npy
        ├── layers.25.feed_forward.w2.weight.npy
        ├── layers.25.feed_forward.w3.weight.npy
        ├── layers.25.ffn_norm.weight.npy
        ├── layers.26.attention_norm.weight.npy
        ├── layers.26.attention.wk.weight.npy
        ├── layers.26.attention.wo.weight.npy
        ├── layers.26.attention.wq.weight.npy
        ├── layers.26.attention.wv.weight.npy
        ├── layers.26.feed_forward.w1.weight.npy
        ├── layers.26.feed_forward.w2.weight.npy
        ├── layers.26.feed_forward.w3.weight.npy
        ├── layers.26.ffn_norm.weight.npy
        ├── layers.27.attention_norm.weight.npy
        ├── layers.27.attention.wk.weight.npy
        ├── layers.27.attention.wo.weight.npy
        ├── layers.27.attention.wq.weight.npy
        ├── layers.27.attention.wv.weight.npy
        ├── layers.27.feed_forward.w1.weight.npy
        ├── layers.27.feed_forward.w2.weight.npy
        ├── layers.27.feed_forward.w3.weight.npy
        ├── layers.27.ffn_norm.weight.npy
        ├── layers.28.attention_norm.weight.npy
        ├── layers.28.attention.wk.weight.npy
        ├── layers.28.attention.wo.weight.npy
        ├── layers.28.attention.wq.weight.npy
        ├── layers.28.attention.wv.weight.npy
        ├── layers.28.feed_forward.w1.weight.npy
        ├── layers.28.feed_forward.w2.weight.npy
        ├── layers.28.feed_forward.w3.weight.npy
        ├── layers.28.ffn_norm.weight.npy
        ├── layers.29.attention_norm.weight.npy
        ├── layers.29.attention.wk.weight.npy
        ├── layers.29.attention.wo.weight.npy
        ├── layers.29.attention.wq.weight.npy
        ├── layers.29.attention.wv.weight.npy
        ├── layers.29.feed_forward.w1.weight.npy
        ├── layers.29.feed_forward.w2.weight.npy
        ├── layers.29.feed_forward.w3.weight.npy
        ├── layers.29.ffn_norm.weight.npy
        ├── layers.2.attention_norm.weight.npy
        ├── layers.2.attention.wk.weight.npy
        ├── layers.2.attention.wo.weight.npy
        ├── layers.2.attention.wq.weight.npy
        ├── layers.2.attention.wv.weight.npy
        ├── layers.2.feed_forward.w1.weight.npy
        ├── layers.2.feed_forward.w2.weight.npy
        ├── layers.2.feed_forward.w3.weight.npy
        ├── layers.2.ffn_norm.weight.npy
        ├── layers.30.attention_norm.weight.npy
        ├── layers.30.attention.wk.weight.npy
        ├── layers.30.attention.wo.weight.npy
        ├── layers.30.attention.wq.weight.npy
        ├── layers.30.attention.wv.weight.npy
        ├── layers.30.feed_forward.w1.weight.npy
        ├── layers.30.feed_forward.w2.weight.npy
        ├── layers.30.feed_forward.w3.weight.npy
        ├── layers.30.ffn_norm.weight.npy
        ├── layers.31.attention_norm.weight.npy
        ├── layers.31.attention.wk.weight.npy
        ├── layers.31.attention.wo.weight.npy
        ├── layers.31.attention.wq.weight.npy
        ├── layers.31.attention.wv.weight.npy
        ├── layers.31.feed_forward.w1.weight.npy
        ├── layers.31.feed_forward.w2.weight.npy
        ├── layers.31.feed_forward.w3.weight.npy
        ├── layers.31.ffn_norm.weight.npy
        ├── layers.32.attention_norm.weight.npy
        ├── layers.32.attention.wk.weight.npy
        ├── layers.32.attention.wo.weight.npy
        ├── layers.32.attention.wq.weight.npy
        ├── layers.32.attention.wv.weight.npy
        ├── layers.32.feed_forward.w1.weight.npy
        ├── layers.32.feed_forward.w2.weight.npy
        ├── layers.32.feed_forward.w3.weight.npy
        ├── layers.32.ffn_norm.weight.npy
        ├── layers.33.attention_norm.weight.npy
        ├── layers.33.attention.wk.weight.npy
        ├── layers.33.attention.wo.weight.npy
        ├── layers.33.attention.wq.weight.npy
        ├── layers.33.attention.wv.weight.npy
        ├── layers.33.feed_forward.w1.weight.npy
        ├── layers.33.feed_forward.w2.weight.npy
        ├── layers.33.feed_forward.w3.weight.npy
        ├── layers.33.ffn_norm.weight.npy
        ├── layers.34.attention_norm.weight.npy
        ├── layers.34.attention.wk.weight.npy
        ├── layers.34.attention.wo.weight.npy
        ├── layers.34.attention.wq.weight.npy
        ├── layers.34.attention.wv.weight.npy
        ├── layers.34.feed_forward.w1.weight.npy
        ├── layers.34.feed_forward.w2.weight.npy
        ├── layers.34.feed_forward.w3.weight.npy
        ├── layers.34.ffn_norm.weight.npy
        ├── layers.35.attention_norm.weight.npy
        ├── layers.35.attention.wk.weight.npy
        ├── layers.35.attention.wo.weight.npy
        ├── layers.35.attention.wq.weight.npy
        ├── layers.35.attention.wv.weight.npy
        ├── layers.35.feed_forward.w1.weight.npy
        ├── layers.35.feed_forward.w2.weight.npy
        ├── layers.35.feed_forward.w3.weight.npy
        ├── layers.35.ffn_norm.weight.npy
        ├── layers.36.attention_norm.weight.npy
        ├── layers.36.attention.wk.weight.npy
        ├── layers.36.attention.wo.weight.npy
        ├── layers.36.attention.wq.weight.npy
        ├── layers.36.attention.wv.weight.npy
        ├── layers.36.feed_forward.w1.weight.npy
        ├── layers.36.feed_forward.w2.weight.npy
        ├── layers.36.feed_forward.w3.weight.npy
        ├── layers.36.ffn_norm.weight.npy
        ├── layers.37.attention_norm.weight.npy
        ├── layers.37.attention.wk.weight.npy
        ├── layers.37.attention.wo.weight.npy
        ├── layers.37.attention.wq.weight.npy
        ├── layers.37.attention.wv.weight.npy
        ├── layers.37.feed_forward.w1.weight.npy
        ├── layers.37.feed_forward.w2.weight.npy
        ├── layers.37.feed_forward.w3.weight.npy
        ├── layers.37.ffn_norm.weight.npy
        ├── layers.38.attention_norm.weight.npy
        ├── layers.38.attention.wk.weight.npy
        ├── layers.38.attention.wo.weight.npy
        ├── layers.38.attention.wq.weight.npy
        ├── layers.38.attention.wv.weight.npy
        ├── layers.38.feed_forward.w1.weight.npy
        ├── layers.38.feed_forward.w2.weight.npy
        ├── layers.38.feed_forward.w3.weight.npy
        ├── layers.38.ffn_norm.weight.npy
        ├── layers.39.attention_norm.weight.npy
        ├── layers.39.attention.wk.weight.npy
        ├── layers.39.attention.wo.weight.npy
        ├── layers.39.attention.wq.weight.npy
        ├── layers.39.attention.wv.weight.npy
        ├── layers.39.feed_forward.w1.weight.npy
        ├── layers.39.feed_forward.w2.weight.npy
        ├── layers.39.feed_forward.w3.weight.npy
        ├── layers.39.ffn_norm.weight.npy
        ├── layers.3.attention_norm.weight.npy
        ├── layers.3.attention.wk.weight.npy
        ├── layers.3.attention.wo.weight.npy
        ├── layers.3.attention.wq.weight.npy
        ├── layers.3.attention.wv.weight.npy
        ├── layers.3.feed_forward.w1.weight.npy
        ├── layers.3.feed_forward.w2.weight.npy
        ├── layers.3.feed_forward.w3.weight.npy
        ├── layers.3.ffn_norm.weight.npy
        ├── layers.4.attention_norm.weight.npy
        ├── layers.4.attention.wk.weight.npy
        ├── layers.4.attention.wo.weight.npy
        ├── layers.4.attention.wq.weight.npy
        ├── layers.4.attention.wv.weight.npy
        ├── layers.4.feed_forward.w1.weight.npy
        ├── layers.4.feed_forward.w2.weight.npy
        ├── layers.4.feed_forward.w3.weight.npy
        ├── layers.4.ffn_norm.weight.npy
        ├── layers.5.attention_norm.weight.npy
        ├── layers.5.attention.wk.weight.npy
        ├── layers.5.attention.wo.weight.npy
        ├── layers.5.attention.wq.weight.npy
        ├── layers.5.attention.wv.weight.npy
        ├── layers.5.feed_forward.w1.weight.npy
        ├── layers.5.feed_forward.w2.weight.npy
        ├── layers.5.feed_forward.w3.weight.npy
        ├── layers.5.ffn_norm.weight.npy
        ├── layers.6.attention_norm.weight.npy
        ├── layers.6.attention.wk.weight.npy
        ├── layers.6.attention.wo.weight.npy
        ├── layers.6.attention.wq.weight.npy
        ├── layers.6.attention.wv.weight.npy
        ├── layers.6.feed_forward.w1.weight.npy
        ├── layers.6.feed_forward.w2.weight.npy
        ├── layers.6.feed_forward.w3.weight.npy
        ├── layers.6.ffn_norm.weight.npy
        ├── layers.7.attention_norm.weight.npy
        ├── layers.7.attention.wk.weight.npy
        ├── layers.7.attention.wo.weight.npy
        ├── layers.7.attention.wq.weight.npy
        ├── layers.7.attention.wv.weight.npy
        ├── layers.7.feed_forward.w1.weight.npy
        ├── layers.7.feed_forward.w2.weight.npy
        ├── layers.7.feed_forward.w3.weight.npy
        ├── layers.7.ffn_norm.weight.npy
        ├── layers.8.attention_norm.weight.npy
        ├── layers.8.attention.wk.weight.npy
        ├── layers.8.attention.wo.weight.npy
        ├── layers.8.attention.wq.weight.npy
        ├── layers.8.attention.wv.weight.npy
        ├── layers.8.feed_forward.w1.weight.npy
        ├── layers.8.feed_forward.w2.weight.npy
        ├── layers.8.feed_forward.w3.weight.npy
        ├── layers.8.ffn_norm.weight.npy
        ├── layers.9.attention_norm.weight.npy
        ├── layers.9.attention.wk.weight.npy
        ├── layers.9.attention.wo.weight.npy
        ├── layers.9.attention.wq.weight.npy
        ├── layers.9.attention.wv.weight.npy
        ├── layers.9.feed_forward.w1.weight.npy
        ├── layers.9.feed_forward.w2.weight.npy
        ├── layers.9.feed_forward.w3.weight.npy
        ├── layers.9.ffn_norm.weight.npy
        ├── norm.weight.npy
        ├── output.weight.npy
        ├── rope.freqs.npy
        └── tok_embeddings.weight.npy


```

