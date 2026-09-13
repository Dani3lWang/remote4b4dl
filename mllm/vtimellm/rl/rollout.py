"""Batched sampling plus the sampling-time log-probability (path A).

`outputs.scores` holds the logits *after* the logit processors ran, so with
top_p=1.0 they are exactly logits/temperature and `log_softmax(scores)` is the
log-probability of the distribution that actually picked the token. That is the
quantity GRPO's importance ratio needs; logprob.forward_sequence divides by the
same temperature so the two sides match.
"""

import torch

from .model_utils import stop_string
from .logprob import left_pad


def completion_mask(completion_ids, eos_id):
    """True up to and including the first EOS, False after.

    Deriving the mask from EOS rather than from `!= pad_token_id` is deliberate:
    the pad token is <unk> (id 0) here, and a legitimately sampled <unk> would
    otherwise truncate the completion.
    """
    mask = torch.ones_like(completion_ids, dtype=torch.bool)
    for b in range(completion_ids.shape[0]):
        hit = (completion_ids[b] == eos_id).nonzero()
        if hit.numel():
            mask[b, int(hit[0]) + 1:] = False
    return mask


def decode_texts(tokenizer, completion_ids, stop_str):
    """Same post-processing as inference.py:70-74, so reward sees eval-like text."""
    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    out = []
    for t in texts:
        t = t.strip()
        if stop_str and t.endswith(stop_str):
            t = t[:-len(stop_str)].strip()
        out.append(t)
    return out


@torch.no_grad()
def generate_batch(model, tokenizer, prompt_ids_list, images, *,
                   do_sample=True, temperature=0.9, top_p=1.0,
                   max_new_tokens=64, seed=None):
    """One generate call for a whole batch; every row must share N (see data.py).

    Returns completion ids, per-token logp of the sampled tokens, the EOS mask
    and the decoded texts.
    """
    if seed is not None:
        torch.manual_seed(seed)
    device = next(model.parameters()).device
    n_visual = {int(im.shape[0]) for im in images}
    if len(n_visual) != 1:
        raise ValueError(
            f'all rows in a generate batch must have the same visual-token '
            f'count, got {sorted(n_visual)}')

    pad_id = tokenizer.pad_token_id
    ids, mask, _ = left_pad([p.to(torch.long) for p in prompt_ids_list], pad_id)
    input_len = ids.shape[1]
    # features arrive fp16 (evaluation.test_b4dl.load_features); mm_projector
    # needs the actor's dtype, which is bf16 for the RL processes.
    dtype = next(model.parameters()).dtype
    imgs = [im.to(device=device, dtype=dtype) for im in images]

    stop_str = stop_string()
    gen_kwargs = dict(
        input_ids=ids.to(device),
        attention_mask=mask.to(device),
        images=imgs,
        do_sample=do_sample,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if do_sample:
        gen_kwargs['temperature'] = temperature
        gen_kwargs['top_p'] = top_p
    out = model.generate(**gen_kwargs)

    completion = out.sequences[:, input_len:]
    scores = torch.stack(out.scores, dim=1)          # (B, L_gen, V)
    log_probs = torch.log_softmax(scores.float(), dim=-1)
    logp = log_probs.gather(2, completion[:, :, None]).squeeze(2).cpu()
    keep = completion_mask(completion.cpu(), tokenizer.eos_token_id)
    texts = decode_texts(tokenizer, completion, stop_str)

    return {'completion_ids': completion.cpu(),
            'logp': logp * keep,
            'mask': keep,
            'texts': texts,
            'input_len': input_len,
            'stop_str': stop_str}
