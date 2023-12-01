# LLM Sampler

Install with:

```bash
pip install llm_sampler
```

[![Downloads](https://static.pepy.tech/badge/llm_sampler/month)](https://pepy.tech/project/llm_sampler)
[![PyPi version](https://badgen.net/pypi/v/llm_sampler/)](https://pypi.com/project/llm_sampler)
[![PyPI license](https://img.shields.io/pypi/l/llm_sampler.svg)](https://pypi.python.org/pypi/llm_sampler/)

[![Read the Docs Badge](https://img.shields.io/badge/Read%20the%20Docs-8CA1AF?logo=readthedocs&logoColor=fff&style=for-the-badge)](https://microlib.org/llm_sampler.html)

`llm_sampler` allows you to sample from any LLM.

It is a collection of various sampling techniques found online.

You can use it with any model from `llm_microlibs` and even Huggingface Transformers, mistral, remotely called models.

It also allows you get probability scores for sequences given by the user.

For example, if you supply the input:
Input: `The sentiment of the sentence 'I loved it' is `
- Option 0: `positive`
- Option 1: `negative`

This lib will return the probabilities for the options. 
In that sense, `llm_sampler` can be used as a zero-shot classifier.

## Sampling overview

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

## Example - Huggingface pipeline

```python
import torch
import transformers
from transformers import AutoTokenizer
from llm_sampler import sample
from tqdm import tqdm

model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # device_map="auto",
    device=torch.device("cuda")
)

input_text = "Magnus Carlsen had won the World "
input_ids = pipeline.tokenizer(input_text, padding=False, add_special_tokens=False, return_tensors="pt")
input_ids = input_ids.to(torch.device("cuda"))["input_ids"]

generator = sample(
    forward_func=lambda x: pipeline.model(input_ids=x).logits,
    input_ids=input_ids,
    max_new_tokens=2,
    temperature=0.001
)
result_tokens = []
for token in tqdm(generator):
    int_token = token.cpu().item()
    result_tokens.append(int_token)
decoded = pipeline.tokenizer.decode(result_tokens, skip_special_tokens=True)
```

## Example - score batch

Sample from an LLM with multiple choice:

```python
from llm_sampler import score_batch

# Initializes the forward_func.
# This could be any function that returns logits when given input tokens
# For example, Hugggingface Models, LLaMa, Falcon, etc.
forward_func = load_model()

scores = score_batch(
    forward_func=forward_func,
    input_ids=tokenize_input("The sentiment of the sentence 'I loved it' is '"),
    all_continuation_ids=[
        tokenize_input("positive sentiment"),
        tokenize_input("negative"),
        tokenize_input("neutral"),
    ]
)

# scores is now 
# tensor([[-1.0078, -2.5625],
#         [ 0.6914, -7.0312],
#         [-4.4062, -7.9688]], dtype=torch.bfloat16)
# 
```

