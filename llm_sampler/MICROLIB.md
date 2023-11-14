# llm_sampler

Here you can find a full list of the things you can do with `llm_sampler`.

## Contents

1. [Sample](#sample)
2. [Closed sampling](#dump-safetensors-to-a-directory)

###  Sample

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

### Closed sampling

Closed sampling means that you restrict the output of the model to several possible predefined outputs.