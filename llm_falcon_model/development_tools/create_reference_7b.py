import torch
import transformers
from transformers import AutoTokenizer


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


def main():
    pipeline = load_huggingface_pipeline()
    input_text = "Magnus Carlsen won the World "
    input_ids = torch.tensor(pipeline.tokenizer.encode(input_text)).unsqueeze(0).cuda()
    reference = {
        'input_text': input_text
    }
    with torch.inference_mode():
        logits_hf = pipeline.model(input_ids).logits.cpu()
        reference['logits'] = logits_hf
    sequences = pipeline(
        input_text,
        max_new_tokens=100,
        do_sample=False,
        # top_k=10,
        num_return_sequences=1,
        eos_token_id=pipeline.tokenizer.eos_token_id,
    )
    reference['generated'] = sequences[0]['generated_text']
    torch.save(reference, './reference_7b.pth')


if __name__ == '__main__':
    main()
