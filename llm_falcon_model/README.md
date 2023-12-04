# LLM Falcon model


`llm_falcon_model` allows you to run a part of a Falcon model as a standalone PyTorch module.
This enables you to run in distributed mode, using even old GPUs with less memory.

It only contains code needed for inference.
The only dependencies are `torch`, `tokenizers`, `einops` and `llm_sepweight`.

The original implementation is available [here](https://huggingface.co/tiiuae).


Use it when you cannot fit the whole Falcon model into memory. If you have multiple
old GPUs with less memory, you can run different parts of the Falcon model on each of them and when
you make them communicate (using for example `llm_partial_run`), you can run the full model on multiple
heterogeneous hosts. For example, if you have 4 old gaming PCs with a 3090 card (~6000$), you can run Falcon 40B
real-time (5-6 tokens/s)

You can also use it when you want to run Falcon on a large number of inputs and have insufficient memory for the model.
You can serialize the intermediary results for all inputs and then continue with the next layers

Install with:

```bash
pip install llm_falcon_model
```

[![Downloads](https://static.pepy.tech/badge/llm_falcon_model/month)](https://pepy.tech/project/llm_falcon_model)
[![PyPi version](https://badgen.net/pypi/v/llm_falcon_model/)](https://pypi.com/project/llm_falcon_model)
[![PyPI license](https://img.shields.io/pypi/l/llm_falcon_model.svg)](https://pypi.python.org/pypi/llm_falcon_model/)

## Overview

The most important methods of this microlib are:
1. `llm_falcon_model.load_tokenizer()` - which loads an instance of the `Tokenizer` for the models
2. `llm_falcon_model.init_part(model_name, spec, device)` - which creates a part of a Falcon model, by a given name (`7b`, `40b` or `180b`),
part specification (which layers you want to load, see [sepweight part spec](https://microlib.org/llm_sepweight.html#load-state-dict-needed-for-a-part-of-the-llm))
and a PyTorch device.
3. `llm_falcon_model.generate` - which allows you to generate text based on a prompt.
4. `llm_falcon_model.score_batch` - which allows you to score a bunch of possible continuations based on a prompt.

## Quick example


```python
import torch
import llm_falcon_model

tokenizer = llm_falcon_model.load_tokenizer()

separated_weights_path = '<PATH TO SEPARATED WEIGHTS>'

model = llm_falcon_model.init_part(
    model_name='40b',
    spec='b 0-12', # Load begin and layers 0 to 12
    device='cuda:0'
)

input_text = "The world chess champion Magnus Carlsen"
input_ids = tokenizer.encode(input_text).ids
batch = torch.tensor(input_ids).unsqueeze(0)
x = model(batch)

# x is now the result after end layer 12, shaped:
# torch.Size([1, 7, 8192])
```
