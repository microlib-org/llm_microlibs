# llm_microlibs

`llm_microlibs` consists of several small [microlibs](http://microlib.org/), which
enable you to run **parts** of a bigger LLM (large language model).

The parts are being run in sequential manner, and the time during which a node is not busy with computation
can be used to load future layers into the GPU memory, which allows extremely fast run of full, unquantized LLMs on
even on consumer grade hardware.

Every part is just a standard PyTorch `nn.Module`.

A similar time per token currently cannot be achieved with libraries such as `accelerate` or PyTorch distributed,
because they only preallocate once on the GPU and do not reload future layers while current layer is idle.

For example, if you have multiple old GPUs with less memory, you can run different parts of the LLM on each of them and
when you make them communicate, you can run the full model on multiple heterogeneous hosts.
For example, if you have 4 old gaming PCs with a 3090 card (~6000$), you can run Falcon 40B real-time (5-6 tokens/s).

All microlibs have minimal third-party dependencies and no new abstractions in the public API, which means
you can combine them with any framwework like Huggingface Transformers, xformers, etc.

## List of microlibs

1. [llm_sampler](#llm-sampler)
2. [llm_sepweight](#llm-sepweight)
3. [llm_partial_run](#llm-partial-run)
4. [llm_falcon_model](#llm-falcon-model)


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

### LLM Falcon model

`llm_falcon_model` allows you to run a part of a Falcon model as a standalone PyTorch module.
This enables you to run in distributed mode, using even old GPUs with less memory.

[Read more](./llm_falcon_model/README.md)


## Future work

- [ ] Vectorized version of `sample_multiple_choice` in `llm_sampler`
- [ ] Distribute weights in a `sepweight` format
- [ ] Release `llm_llama2_model`
- [ ] Release `llm_qwen_model`
- [ ] Release `llm_mistral_model` and future bigger models by Mistral
- [ ] Implement a microlib for batched predict
- [ ] Make a write-up of an example distributed run.

... and many more!

Thank you for your interest, if you like our work, please consider leaving a star and sharing it with your friends.