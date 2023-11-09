# LLM partial run

The `llm_partial_run` allows you to run a part of an LLM using `socket_rpc` to communicate with other nodes.

It also allows you to significantly increase the amount of GPU memory available, by keeping the weights of
future layers into CPU memory and loading them into the GPU while the other nodes are computing.

This allows very high GPU utilization for LLMs which do not fit into the total GPU memory.


Install with:

```bash
pip install llm_partial_run
```

[![Downloads](https://static.pepy.tech/badge/llm_partial_run/month)](https://pepy.tech/project/llm_partial_run)
[![PyPi version](https://badgen.net/pypi/v/llm_partial_run/)](https://pypi.com/project/llm_partial_run)
[![PyPI license](https://img.shields.io/pypi/l/llm_partial_run.svg)](https://pypi.python.org/pypi/llm_partial_run/)


## Contents

This microlib allows you to run a part of a large language model (LLM).


Why would you do this? If you have let's say 2 machine with 2 4090 cards each, you'll be able
to run models which require up to 96GB GPU memory.

While one downside of this approach is that there is additional overhead for sending data over the network,
you can use this time to reload weights of future weights.
For example, let's say you want to run Falcon40B, which requires 78GB of GPU memory and has 60 layers.
Let's say you only have two old gaming machines, each with a 3090 GPU card. Then you have in total just 48GB GPU memory.
What you can do is load is:
* On host 0, load layers 0 to 15 and 30 to 45 in the CPU memory.
* On host 1, load layers 15 to 30 and 45 to 60 in the CPU memory.

The transfer from CPU to GPU takes roughly about a second, while for big prompts one
forward pass through the 3090s takes about 600ms.
With `llm_partial_run` you can do the forward pass for layers 0 to 15 on host 0 and send it to the next host,
while on host 0 it immediately begins to transfer layers 30 to 45 to the GPU.
This way, while host 1 is doing computations, host 0 is getting prepared.
When host 1 is done, it sends the result to host 0, so that layers 30 to 45 can make a forward pass,
and in the meantime, host 1 starts transfering layers 45 to 60 to the GPU.
When host 0 is done with layers 30 to 45, it sends the result to host 1, which then
computes the final results.
This would allow you to sample tokens at roughly 2s/token even for big prompts with very cheap hardware.
What is even cooler that you make it faster by buying more hardware, for example if you buy
two more machines of the same kind you can easily compose them with the ones you already have and you will get 5-6 tokens per second.

