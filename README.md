# llm_microlibs

`llm_microlibs` consists of several small [microlibs](http://microlib.org/), which
enable you to run **parts** of a bigger LLM (large language model).

The parts are being run in sequential manner, and the time during which a node is not busy with computation
can be used to load future layers into the GPU memory, which allows decent run of full, unquantized LLMs even
on consumer grade hardware.

Every part is just a standard PyTorch `nn.Module`.

A similar time per token currently cannot be achieved with libraries such as `accelerate` or PyTorch distributed,
because they only preallocate once on the GPU and do not reload future layers while current layer is idle.

### Example: multiple, possibly heterogeneous GPUs

For example, if you have multiple old GPUs with less memory, you can run different parts of the LLM on each of them and
when you make them communicate, you can run the full model on multiple heterogeneous hosts.
For example, if you have 4 old gaming PCs with a 3090 card (~6000$), you can run 40B models real-time (5-6 tokens/s).

### Example: fitting 3 7b models on 2 24GB GPUs

Most 7b models require around 15GB of memory. If you have 2 GPUs, each of which have 24GB, you can load three
7b models into your two GPUs, by loading three halves of each models on each GPU.

All microlibs have minimal third-party dependencies and few abstractions in the public API, which means
you can combine them with any framework like Huggingface Transformers, xformers, etc.

## List of microlibs

1. [llm_sampler](#llm-sampler)
2. [llm_sepweight](#llm-sepweight)
3. [llm_partial_run](#llm-partial-run)
4. [llm_falcon_model](#llm-falcon-model)


### LLM sampler

Install with:

```bash
pip install llm_sampler
```

[![Downloads](https://static.pepy.tech/badge/llm_sampler/month)](https://pepy.tech/project/llm_sampler)
[![PyPi version](https://badgen.net/pypi/v/llm_sampler/)](https://pypi.com/project/llm_sampler)
[![PyPI license](https://img.shields.io/pypi/l/llm_sampler.svg)](https://pypi.python.org/pypi/llm_sampler/)

`llm_sampler` allows you to sample from any LLM.
It accepts a `forward_func` as a parameter, which could be any Python function, which accepts `input_ids` tensor and
outputs `logits` tensor.

You can use it with any model from `llm_microlibs` and even Huggingface Transformers, mistral, remotely called models.

It also allows you get probability scores for sequences given by the user,
which can be used to answer closed-form questions by LLM models.

[Read more](./llm_sampler/README.md)

### LLM sepweight

Install with:

```bash
pip install llm_sepweight
```

[![Downloads](https://static.pepy.tech/badge/llm_sepweight/month)](https://pepy.tech/project/llm_sepweight)
[![PyPi version](https://badgen.net/pypi/v/llm_sepweight/)](https://pypi.com/project/llm_sepweight)
[![PyPI license](https://img.shields.io/pypi/l/llm_sepweight.svg)](https://pypi.python.org/pypi/llm_sepweight/)


The `llm_sepweight` microlib is designed to manage the weights of large language models (LLMs) by organizing them into directories.

It allows you to store and distribute the weights of an LLM as normal files for each layer.

[Read more](./llm_sepweight/README.md)

### LLM partial run

Install with:

```bash
pip install llm_partial_run
```

[![Downloads](https://static.pepy.tech/badge/llm_partial_run/month)](https://pepy.tech/project/llm_partial_run)
[![PyPi version](https://badgen.net/pypi/v/llm_partial_run/)](https://pypi.com/project/llm_partial_run)
[![PyPI license](https://img.shields.io/pypi/l/llm_partial_run.svg)](https://pypi.python.org/pypi/llm_partial_run/)


The `llm_partial_run` allows you to run a part of an LLM using `socket_rpc` to communicate with other nodes.

It also allows you to significantly increase the amount of GPU memory available, by keeping the weights of
future layers into CPU memory and loading them into the GPU while the other nodes are computing.

This allows very high GPU utilization for LLMs which do not fit into the total GPU memory.

[Read more](./llm_partial_run/README.md)

### LLM Falcon model

Install with:

```bash
pip install llm_falcon_model
```

[![Downloads](https://static.pepy.tech/badge/llm_falcon_model/month)](https://pepy.tech/project/llm_falcon_model)
[![PyPi version](https://badgen.net/pypi/v/llm_falcon_model/)](https://pypi.com/project/llm_falcon_model)
[![PyPI license](https://img.shields.io/pypi/l/llm_falcon_model.svg)](https://pypi.python.org/pypi/llm_falcon_model/)


`llm_falcon_model` allows you to run a part of a Falcon model as a standalone PyTorch module.
This enables you to run in distributed mode, using even old GPUs with less memory.

[Read more](./llm_falcon_model/README.md)


## Future work

- [ ] Vectorized version of `sample_multiple_choice` in `llm_sampler`
- [ ] Distribute weights in a `sepweight` format
- [ ] Release `llm_llama2_model`
- [ ] Release `llm_qwen_model`
- [ ] Release `llm_goliath_model`
- [ ] Release `llm_yi_model`
- [ ] Release `llm_mistral_model` and future bigger models by Mistral
- [ ] Integrate `deepseek` models
- [ ] Make a write-up of an example distributed run.

... and many more!

Thank you for your interest, if you like our work, please consider leaving a star and sharing it with your friends.