# LLM Sampler

Install with:

```bash
pip install llm_sampler
```

[![Downloads](https://static.pepy.tech/badge/llm_sampler/month)](https://pepy.tech/project/llm_sampler)
[![PyPi version](https://badgen.net/pypi/v/llm_sampler/)](https://pypi.com/project/llm_sampler)
[![PyPI license](https://img.shields.io/pypi/l/llm_sampler.svg)](https://pypi.python.org/pypi/llm_sampler/)


## Contents

1. [Quick example](#quick-example)
2. [What is it?](#what-is-it)

## Quick example

Sample from an LLM with temperature:

```python
import torch
from llm_sampler import sample

# Initializes the forward_func. 
# This could be any function that returns logits when given input tokens 
# For example, Hugggingface Models, LLaMa, Falcon, etc.
forward_func = load_model() 
input_ids = tokenize_input("Magnus Carlsen had won the World ") # Tokenize the input
max_new_tokens = 10  # Number of new tokens to generate

generated_tokens = sample(
    forward_func=forward_func, 
    input_ids=input_ids, 
    max_new_tokens=max_new_tokens, 
    temperature=0.6, 
    warp_top_k=10
)
for next_token in generated_tokens:
    print("Next token:", next_token)

```

Sample from an LLM with multiple choice:

```python
from llm_sampler import sample_multiple_choice

# Initializes the forward_func. 
# This could be any function that returns logits when given input tokens 
# For example, Hugggingface Models, LLaMa, Falcon, etc.
forward_func = load_model() 

generator = sample_multiple_choice(
    forward_func=forward_func,
    input_ids=tokenize_input("The sentiment of the sentence 'I loved it' is '"),
    all_continuation_ids=[
        tokenize_input("positive"),
        tokenize_input("negative")
    ]
)
raw_seqs = list(generator)

# raw_seqs is now [tensor([0.2031], dtype=torch.bfloat16), tensor([-1.5781], dtype=torch.bfloat16)]
```

## What is it

`llm_sampler` is a microlib which allows you to sample from an LLM, or give the probability scores for 
sequences given by the user. 

For example, if you supply the input:
Input: `The sentiment of the sentence 'I loved it' is `
- Option 0: `positive`
- Option 1: `negative`

This lib will return the probabilities for the options. 
In that sense, `llm_sampler` can be used as a zero-shot classifier.