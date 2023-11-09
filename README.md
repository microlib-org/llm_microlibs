# llm_microlibs

`llm_microlibs` consists of several small [microlibs](http://microlib.org/), which
enable you to run **parts** of a bigger LLM (large language model).

For example, using the `llm_falcon_model` microlib, you can run Falcon40B model layers
0 to 10 as a standard PyTorch module on one GPU, layers 10 to 20 on another (even another host), etc.
All those parts of the model are just normal PyTorch `nn.Module`s, which allows great flexibility.

## List of microlibs

1. [llm_sampler](#llm-sampler)
2. [llm_sepweight](#llm-sepweights)


### LLM sampler

`llm_sampler` allows you to sample from any LLM.
It accepts a `forward_func` as a parameter, which could be any Python function, which accepts `input_ids` tensor and
outputs `logits` tensor.

It also allows you get probability scores for sequences given by the user.

[Read more](./llm_sampler/README.md)

### LLM sepweight

[Read more](./llm_sepweight/README.md)

### Why do we need it

Why would you do this? If you have let's say 2 machine with 2 4090 cards each, you'll be able
to run models which require up to 96GB GPU memory.

While own downside of this approach is that there is additional overhead for sending data over the network,
you can use this time to reload weights of future weights.
For example, let's say you want to run Falcon40B, which requires 78GB of GPU memory and has 60 layers.
Let's say you only have two old gaming machines, each with a 3090 GPU card. Then you have in total just 48GB GPU memory.
What you can do is load is:
* On host 0, load layers 0 to 15 and 30 to 45 in the CPU memory.
* On host 1, load layers 15 to 30 and 45 to 60 in the CPU memory.

The transfer from CPU to GPU takes roughly about a second, while for big prompts one
forward pass through the 3090s takes about 600ms.
With `llm_microlibs` you can do the forward pass for layers 0 to 15 on host 0 and send it to the next host,
while on host 0 it immediately begins to transfer layers 30 to 45 to the GPU.
This way, while host 1 is doing computations, host 0 is getting prepared.
When host 1 is done, it sends the result to host 0, so that layers 30 to 45 can make a forward pass,
and in the meantime, host 1 starts transfering layers 45 to 60 to the GPU.
When host 0 is done with layers 30 to 45, it sends the result to host 1, which then
computes the final results.
This would allow you to sample tokens at roughly 2s/token even for big prompts with very cheap hardware.
What is even cooler that you make it faster by buying more hardware, for example if you buy
two more machines of the same kind you can easily compose them with the ones you already have and you will get 5-6 tokens per second.

All microlibs have minimal third-party dependencies.