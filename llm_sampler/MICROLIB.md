# llm_sampler

`llm_sampler` allows you to sample from any LLM by given logits.

Here you can find a full list of the things you can do with `llm_sampler`.

## Contents

1. [Sample (Huggingface)](#sample-huggingface)
2. [Sample (GPT-fast)](#sample-gpt-fast)

###  Sample (Huggingface)


```python
import llm_sampler

outputs = ...#tensor should have 3 dimensions: batch, sequence length, vocabulary

next_tokens = llm_sampler.sample_huggingface(
    outputs=outputs,
    temperature=0.7,
    top_k=90
)
```

### Sample (GPT-fast)


```python
import llm_sampler

outputs = ...#tensor should have 3 dimensions: batch, sequence length, vocabulary

next_tokens = llm_sampler.sample_gpt_fast(
    outputs=outputs,
    temperature=0.7,
    top_k=90
)
```