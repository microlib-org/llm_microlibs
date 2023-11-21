# llm_sampler

Here you can find a full list of the things you can do with `llm_sampler`.

## Contents

1. [Sample](#sample)
2. [Score batch (iterative)](#score-batch-iterative)

###  Sample

Sampling example:

```python
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

To use Huggingface models with `llm_sampler`, just pass `lambda x: pipeline.model(input_ids=x).logits` as 
a `forward_func`.

Here is an example:

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

In general, you can use any `Callable` function, even normal PyTorch modules as a `forward_func`.

### Score batch (iterative)

Closed sampling means that you restrict the output of the model to several possible predefined outputs.
Sample from an LLM with multiple choice:

```python
from llm_sampler import sco

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

# raw_seqs is now [
# tensor([0.2031], dtype=torch.bfloat16), 
# tensor([-1.5781], dtype=torch.bfloat16)
]
```

