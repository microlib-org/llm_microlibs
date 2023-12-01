from time import time

import torch
import transformers
from transformers import AutoTokenizer


def load_huggingface_pipeline():
    model = "tiiuae/falcon-40b"

    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # device_map="auto",
        device=torch.device("cpu")
    )
    return pipeline


def main():
    pipeline = load_huggingface_pipeline()
    input_text = "Magnus Carlsen won the World "
    input_ids = torch.tensor(pipeline.tokenizer.encode(input_text)).unsqueeze(0).cpu()
    with torch.inference_mode():
        start_t = time()
        logits_hf = pipeline.model(input_ids).logits.cpu()
        print(f'Took {time() - start_t}')
        reference = {
            'logits': logits_hf,
            'input_text': input_text
        }
        torch.save(reference, './reference_40b.pth')


if __name__ == '__main__':
    main()
