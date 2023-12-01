import torch


def _top_k_logits_warper(
        scores: torch.FloatTensor,
        top_k: int,
        filter_value: float = -float("Inf")
) -> torch.FloatTensor:
    top_k = min(top_k, scores.size(-1))
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def sample_huggingface(outputs, top_k=None):
    logits = outputs[:, -1, :]
    scores = _top_k_logits_warper(logits, top_k=top_k) if top_k is not None else logits
    scores = scores / temperature if abs(temperature) > 1e-7 else scores
    probs = nn.functional.softmax(scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)