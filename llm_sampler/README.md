# LLM Sampler

## Quick example

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