"""Why path A (generate scores) and path B (full forward) differ, and proof the
gap is dtype rounding rather than an index error.

The GRPO trainer recomputes logp_old / logp_ref / logp_actor ALL with path B
(logprob.forward_sequence), so path A (rollout.generate_batch's scores) is never
used in training -- the decision recorded in smoke_m2 G2.4's [info] line. This
diagnostic justifies that the residual A-vs-B gap is bf16 cache-vs-full-forward
rounding, not a mis-gather:

  * bf16 has an 8-bit mantissa, fp16 an 11-bit one. If the gap is rounding it
    shrinks by ~2**(11-8) = 8x going bf16 -> fp16; an index bug is dtype-
    independent and would stay put.
  * path B == path C (the model's own CE with labels) at every dtype confirms
    path B's gather/shift arithmetic is exact, independent of path A.
  * path B's mean NLL on the tokens path A actually sampled stays near 0.5
    (avg prob ~0.6). A one-position mis-gather would put it near log(vocab)~12.

    python -m vtimellm.rl.diag_logprob_dtype \
        --stage2 checkpoints/...-b3 --data b4dl_dataset/rl_tg_train.jsonl
"""

import argparse

import torch

from . import DEFAULT_B3, DEFAULT_TRAIN_JSON
from .data import FeatureStore, buckets_by_n_visual, load_entries, read_jsonl
from .logprob import forward_sequence, mean_logp
from .model_utils import encode_prompt, load_merged_actor
from .rollout import generate_batch


def _max_gap(a, b, lengths):
    diffs = []
    for k, lc in enumerate(lengths):
        diffs += (a[k, :lc] - b[k, :lc]).abs().tolist()
    if not diffs:
        return 0.0, 0.0, 0
    return max(diffs), sum(diffs) / len(diffs), len(diffs)


def _trim(gen):
    comp = gen['completion_ids']
    return [comp[b][gen['mask'][b]] for b in range(comp.shape[0])]


def score_paths(model, tok, prompts, images, pad, temps, max_new, seed, label):
    """path A (scores) vs path B (gather) vs path C (CE) at each temperature."""
    out = {}
    for T in temps:
        gen = generate_batch(model, tok, prompts, images, do_sample=True,
                             temperature=T, top_p=1.0, max_new_tokens=max_new,
                             seed=seed)
        rows = _trim(gen)
        lengths = [len(r) for r in rows]
        A = gen['logp']
        B = forward_sequence(model, prompts, rows, images, temperature=T,
                             pad_id=pad, to_cpu=True)
        C = forward_sequence(model, prompts, rows, images, temperature=T,
                             with_labels=True, pad_id=pad)
        gmax, gmean, ntok = _max_gap(A, B['logp'], lengths)
        nll_b = -mean_logp(B)
        rel_c = abs(nll_b - C['loss']) / max(abs(C['loss']), 1e-9)
        out[f'T={T}'] = {'A_vs_B_max': gmax, 'A_vs_B_mean': gmean,
                         'n_tokens': ntok, 'mean_nll_B': nll_b,
                         'loss_C': C['loss'], 'B_vs_C_rel': rel_c}
        print(f'  [{label}] T={T}: A-vs-B max {gmax:.4e} mean {gmean:.4e} '
              f'({ntok} tok) | NLL_B {nll_b:.4f} == C {C["loss"]:.4f} '
              f'(rel {rel_c:.2e})', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2', default=DEFAULT_B3)
    ap.add_argument('--data', default=DEFAULT_TRAIN_JSON)
    ap.add_argument('--tasks', nargs='*', default=['time_grounding'])
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--max-new', type=int, default=64)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--skip-fp16', action='store_true')
    a = ap.parse_args()

    if a.data.endswith('.jsonl'):
        entries = read_jsonl(a.data)
    else:
        entries, _ = load_entries(a.data, tasks=tuple(a.tasks) or None)
    buckets = buckets_by_n_visual(entries)
    main_n = max(buckets, key=lambda k: len(buckets[k]))
    pool = buckets[main_n][:a.n]
    store = FeatureStore()
    temps = [1.0, a.temperature]
    print(f'[diag] {len(pool)} entries, N={main_n}; temps {temps}; '
          f'rounding predicts the A-vs-B gap shrinks ~8x bf16->fp16')

    # Tokenisation is dtype-independent: encode the prompts once with the bf16
    # actor's tokenizer, then score the SAME prompts under both dtypes.
    tok, model, _ = load_merged_actor(a.stage2, dtype=torch.bfloat16)
    pad = tok.pad_token_id
    prompts, images = [], []
    for e in pool:
        _, ids, _ = encode_prompt(tok, e['prompt'])
        prompts.append(ids)
        images.append(store.get(e))

    print('[bf16]')
    res = {'bf16': score_paths(model, tok, prompts, images, pad, temps,
                               a.max_new, a.seed, 'bf16')}
    del model

    if not a.skip_fp16:
        print('[fp16]')
        tok16, model16, _ = load_merged_actor(a.stage2, dtype=torch.float16)
        res['fp16'] = score_paths(model16, tok16, prompts, images,
                                  tok16.pad_token_id, temps, a.max_new, a.seed,
                                  'fp16')
        del model16

    print('\n[verdict]')
    for T in temps:
        k = f'T={T}'
        b = res['bf16'][k]['A_vs_B_max']
        if 'fp16' in res:
            f = res['fp16'][k]['A_vs_B_max']
            ratio = b / f if f else float('inf')
            print(f'  {k}: A-vs-B max bf16 {b:.3e} vs fp16 {f:.3e} -> '
                  f'{ratio:.1f}x smaller in fp16 (8->11 mantissa bits predicts '
                  f'~8x). ratio >> 1 => rounding, not an index bug.')
        print(f'  {k}: bf16 path B == path C rel {res["bf16"][k]["B_vs_C_rel"]:.2e}; '
              f'mean NLL_B {res["bf16"][k]["mean_nll_B"]:.3f} (sane ~0.5; a '
              f'mis-gather would sit near log(vocab)~12) => path B aligned.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
