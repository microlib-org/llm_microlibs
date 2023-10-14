# LLM Falcon model

Install with:

```bash
pip install llm_falcon_model
```

[![Downloads](https://static.pepy.tech/badge/llm_falcon_model/month)](https://pepy.tech/project/llm_falcon_model)
[![PyPi version](https://badgen.net/pypi/v/llm_falcon_model/)](https://pypi.com/project/llm_falcon_model)
[![PyPI license](https://img.shields.io/pypi/l/llm_falcon_model.svg)](https://pypi.python.org/pypi/llm_falcon_model/)


## Contents

1. [Quick example](#quick-example)
2. [What is it](#what-is-it)
3. [When to use it](#when-to-use-it)
4. [When not to use it](#when-not-to-use-it)


## Quick example

```python
import torch

from llm_falcon_model import init_part, load_tokenizer

tokenizer = load_tokenizer()

separated_weights_path = '<PATH TO SEPARATED WEIGHTS>'

model = init_part(
    model_name='40b',
    start_layer=0,
    end_layer=12,
    separated_weights_path=separated_weights_path,
    device='cuda:0'
)

input_text = "The world chess champion Magnus Carlsen"
input_ids = tokenizer.encode(input_text).ids
batch = torch.tensor(input_ids).unsqueeze(0)
x = model(batch)

# x is now the result after end layer 12, shaped:
# torch.Size([1, 7, 8192])
```

[Back to Contents](#contents)

## What is it

This microlib allows you to run a part of a Falcon model as a standalone PyTorch module.
This enables you to run in distributed mode, using even old GPUs with less memory.

It only contains code needed for inference.
The only dependencies are `torch`, `tokenizers`, `einops` and `llm_weights_mmap`.

The original implementation is available [here](https://huggingface.co/tiiuae).

[Back to Contents](#contents)

## When to use it

Use it when you cannot fit the whole Falcon model into memory. If you have multiple
old GPUs with less memory, you can run different parts of the Falcon model on each of them and when
you make them communicate (using for example `llm_partial_run`), you can run the full model on multiple
heterogeneous hosts. For example, if you have 4 old gaming PCs with a 3090 card (~6000$), you can run Falcon 40B 
real-time (5-6 tokens/s)

You can also use it when you want to run Falcon on a large number of inputs and have insufficient memory for the model.
You can serialize the intermediary results for all inputs and then continue with the next layers

[Back to Contents](#contents)

## When not to use it

Don't use this library if you want to train or finetune a model, this is just a library
for inference.
