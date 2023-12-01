# LLM Sampler

Install with:

```bash
pip install llm_sampler
```

[![Downloads](https://static.pepy.tech/badge/llm_sampler/month)](https://pepy.tech/project/llm_sampler)
[![PyPi version](https://badgen.net/pypi/v/llm_sampler/)](https://pypi.com/project/llm_sampler)
[![PyPI license](https://img.shields.io/pypi/l/llm_sampler.svg)](https://pypi.python.org/pypi/llm_sampler/)

[![Read the Docs Badge](https://img.shields.io/badge/Read%20the%20Docs-8CA1AF?logo=readthedocs&logoColor=fff&style=for-the-badge)](https://microlib.org/llm_sampler.html)

`llm_sampler` allows you to sample from any LLM by given logits.

It is a collection of various sampling techniques found online.

For now, the methods are:
1. `sample_huggingface` - the one used in [transformers](https://github.com/huggingface/transformers)
2. `sample_gpt_fast` - the one used in [gpt-fast](https://github.com/pytorch-labs/gpt-fast)

It also allows you get probability scores for sequences given by the user.

For example, if you supply the input:
Input: `The sentiment of the sentence 'I loved it' is`
- Option 0: `positive`
- Option 1: `negative`
- Option 2: `unknown`

This lib will help you `create_batch_with_continuations`