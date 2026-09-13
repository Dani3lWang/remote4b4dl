"""Per-token log-probabilities from a full-sequence forward (path B).

The fusion layer replaces the single `<video>` placeholder with N visual
embeddings (arch.py:199-221) and re-pads the batch to the longest *expanded*
row, so the returned logits are not aligned with the input_ids we passed in.
Under left padding every row's completion is flush against the right edge,
which makes the gather index a per-row constant:

    expanded real length  E_b = len(prompt_b) - 1 + N_b + len(completion_b)
    completion token j    ->  expanded index  E_max - L_b^c + j
    its predicting logits ->  index one to the left

`E_max == max_b(E_b)` is asserted on every call, so a wrong layout fails loudly
instead of producing plausible-looking numbers.
"""

import torch

from vtimellm.constants import IMAGE_TOKEN_INDEX

IGNORE_INDEX = -100


def left_pad(rows, pad_id):
    """Left-pad a list of 1-D LongTensors to one (B, L) batch + mask."""
    lengths = [int(r.shape[0]) for r in rows]
    L = max(lengths)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), L), dtype=torch.long)
    for b, r in enumerate(rows):
        ids[b, L - lengths[b]:] = r
        mask[b, L - lengths[b]:] = 1
    return ids, mask, lengths


def expanded_length(prompt_ids, completion_len, n_visual):
    n_img = int((prompt_ids == IMAGE_TOKEN_INDEX).sum())
    if n_img != 1:
        raise ValueError(f'expected exactly one <video> placeholder, got {n_img}')
    return int(prompt_ids.shape[0]) - 1 + n_visual + completion_len


def forward_sequence(model, prompt_ids, completion_ids, images,
                     temperature=1.0, with_labels=False, pad_id=0):
    """logp (B, Lc_max), mask (B, Lc_max) and -- if requested -- the CE loss.

    `temperature` must match the one used at sampling time: transformers applies
    the temperature logit processor before storing `scores`, so the rollout's
    logp describes logits/T. Dividing here too keeps actor, reference and old
    policy on the same tempered distribution.

    `with_labels=True` additionally returns the model's own CE loss over the
    completion tokens only (path C), which is an independent check on the index
    arithmetic above: it must equal the token-weighted mean of -logp at
    temperature 1.0.
    """
    device = next(model.parameters()).device
    rows, labels_rows, exp_lens, comp_lens = [], [], [], []
    for p, c, im in zip(prompt_ids, completion_ids, images):
        p = p.to(dtype=torch.long, device='cpu')
        c = c.to(dtype=torch.long, device='cpu')
        rows.append(torch.cat([p, c]))
        comp_lens.append(int(c.shape[0]))
        exp_lens.append(expanded_length(p, comp_lens[-1], int(im.shape[0])))
        if with_labels:
            ignored = torch.full((int(p.shape[0]),), IGNORE_INDEX, dtype=torch.long)
            labels_rows.append(torch.cat([ignored, c]))
    e_max = max(exp_lens)

    ids, mask, _ = left_pad(rows, pad_id)
    ids = ids.to(device)
    mask = mask.to(device)
    labels = None
    if with_labels:
        labels, _, _ = left_pad(labels_rows, IGNORE_INDEX)
        labels = labels.to(device)
    imgs = [im.to(device=device, dtype=next(model.parameters()).dtype)
            for im in images]

    out = model(input_ids=ids, attention_mask=mask, images=imgs,
                labels=labels, use_cache=False)
    logits = out.logits
    if logits.shape[1] != e_max:
        raise AssertionError(
            f'expanded length mismatch: logits {logits.shape[1]} vs predicted '
            f'{e_max} (per-row {exp_lens})')

    lc_max = max(comp_lens)
    logp = torch.zeros((len(rows), lc_max), dtype=torch.float32)
    keep = torch.zeros((len(rows), lc_max), dtype=torch.bool)
    log_probs = torch.log_softmax(logits.float() / temperature, dim=-1)
    for b, (c, lc) in enumerate(zip(completion_ids, comp_lens)):
        c = c.to(dtype=torch.long, device='cpu')
        start = e_max - lc                      # first completion position
        idx = torch.arange(start - 1, start + lc - 1, device=log_probs.device)
        row = log_probs[b].index_select(0, idx)  # (lc, V)
        logp[b, :lc] = row.gather(1, c.to(row.device)[:, None]).squeeze(1).cpu()
        keep[b, :lc] = True

    result = {'logp': logp, 'mask': keep, 'expanded_lengths': exp_lens,
              'e_max': e_max, 'completion_lengths': comp_lens}
    if with_labels:
        loss = out.loss
        n_tok = int((labels != IGNORE_INDEX).sum())
        result['loss'] = float(loss.detach().cpu())
        result['loss_tokens'] = n_tok
    return result


def mean_logp(result):
    """Token-weighted mean logp over the kept completion positions."""
    logp, mask = result['logp'], result['mask']
    return float((logp * mask).sum() / max(int(mask.sum()), 1))
