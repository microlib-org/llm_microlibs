import torch
import transformers
from transformers import AutoTokenizer

import llm_falcon_model
import llm_sepweight


def load_huggingface_pipeline():
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
    return pipeline


def load_model():
    part = llm_falcon_model.init_part('7b', 'f', 'cuda:1')
    state_dict = llm_sepweight.load('/home/hvrigazov/llm-sepweights/falcon-7b', 'f').to_dict()
    part.load_state_dict(state_dict)
    return part


def main():
    pipeline = load_huggingface_pipeline()
    model = load_model()
    input_ids = torch.tensor(pipeline.tokenizer.encode("Magnus Carlsen won the World ")).unsqueeze(0).cuda()
    with torch.inference_mode():
        logits_hf = pipeline.model(input_ids)
        logits_new = model(input_ids.cuda(1))
        assert torch.allclose(logits_new.cuda(0), logits_hf.logits)
        print('OK')


if __name__ == '__main__':
    main()
