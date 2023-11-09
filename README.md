# llm_microlibs

`llm_microlibs` consists of several small [microlibs](http://microlib.org/), which
enable you to run **parts** of a bigger LLM (large language model).

All microlibs have minimal third-party dependencies and no new abstractions in the public API, which means
you can combine them with any framwework like Huggingface Transformers, xformers, etc.

## List of microlibs

1. [llm_sampler](#llm-sampler)
2. [llm_sepweight](#llm-sepweight)
3. [llm_partial_run](#llm-partial-run)


### LLM sampler

`llm_sampler` allows you to sample from any LLM.
It accepts a `forward_func` as a parameter, which could be any Python function, which accepts `input_ids` tensor and
outputs `logits` tensor.

It also allows you get probability scores for sequences given by the user.

[Read more](./llm_sampler/README.md)

### LLM sepweight

The `llm_sepweight` microlib is designed to manage the weights of large language models (LLMs) by organizing them into directories.

It allows you to store and distribute the weights of an LLM as normal files for each layer.

[Read more](./llm_sepweight/README.md)

### LLM partial run

The `llm_partial_run` allows you to run a part of an LLM using `socket_rpc` to communicate with other nodes.

It also allows you to significantly increase the amount of GPU memory available, by keeping the weights of
future layers into CPU memory and loading them into the GPU while the other nodes are computing.

This allows very high GPU utilization for LLMs which do not fit into the total GPU memory.

[Read more](./llm_partial_run/README.md)

### Why do we need it
