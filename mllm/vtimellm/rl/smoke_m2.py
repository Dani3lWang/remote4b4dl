"""M2 -- rollout path smoke. These gates decide whether a trainer may be written.

    python -m vtimellm.rl.smoke_m2 --stage2 checkpoints/...-b3 \
        --data b4dl_dataset/rl_tg_train.jsonl --batch 8 \
        --out training_logs/smoke_m2.json

G2.2 answer token lengths          -> is max_new_tokens=64 enough
G2.3a inference() vs batch=1 vs batch=B, greedy, equal N
                                     -> the batched path reproduces the frozen
                                        eval path token for token
G2.3b same with mixed N (negative control)
                                     -> must diverge or crash; this is the
                                        evidence for the equal-N rule
G2.4 path A (scores) vs path B (forward+gather) vs path C (labels CE)
                                     -> logp plumbing is correct
G2.5 throughput / peak memory for B in {1,4,8,16}
                                     -> the time budget in the plan is real
G2.6 bf16 vs fp16 greedy agreement   -> the dtype the trainer uses still
    (same prompts, model reloaded)       behaves like the evaluated policy
"""

import argparse
import gc
import json
import os
import time

import torch

from vtimellm.inference import inference

from . import DEFAULT_B3, DEFAULT_TRAIN_JSON
from .data import FeatureStore, buckets_by_n_visual, load_entries, read_jsonl
from .logprob import forward_sequence, mean_logp
from .model_utils import attach_rl_lora, encode_prompt, load_merged_actor
from .reward import parse_failures
from .rollout import generate_batch

RESULTS = []


def gate(name, passed, detail):
    RESULTS.append({'gate': name, 'pass': bool(passed), 'detail': detail})
    print(f'  [{"PASS" if passed else "FAIL"}] {name}: {detail}', flush=True)


def prepare(entries, store, tokenizer, n_items, model=None):
    """Prompt ids + feature tensors for the first n_items entries.

    `inference()` hands the feature tensor to the model without casting it, so
    the cast has to happen here; generate_batch/forward_sequence cast internally.
    """
    dtype = next(model.parameters()).dtype if model is not None else None
    prompts, images = [], []
    for e in entries[:n_items]:
        _, ids, _ = encode_prompt(tokenizer, e['prompt'])
        prompts.append(ids)
        feat = store.get(e)
        images.append(feat if dtype is None else feat.to(dtype))
    return prompts, images


def g2_2_answer_lengths(entries, tokenizer):
    lens = []
    for e in entries:
        lens.append(len(tokenizer(e['gt'])['input_ids']))
    lens.sort()
    q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
    mx = lens[-1]
    ok = q(0.999) <= 48
    gate('G2.2 answer token length', ok,
         f'P50={q(0.5)} P99={q(0.99)} P99.9={q(0.999)} max={mx} -> '
         f'max_new_tokens=64 {"holds" if ok else "TOO SMALL, use " + str(mx + 8)}')
    return {'p50': q(0.5), 'p99': q(0.99), 'p999': q(0.999), 'max': mx}


def g2_3_batch_equivalence(model, tokenizer, entries, store, batch, mixed_n_pool,
                           max_new):
    eq = [e for e in entries[:batch]]
    prompts, images = prepare(eq, store, tokenizer, batch, model)

    ref_texts = [inference(model, im, eq[i]['prompt'], tokenizer,
                           do_sample=False, frame_indices=None)
                 for i, im in enumerate(images)]
    # one single-row generate per item: the unpadded path, which is what
    # inference() exercises and what the all-ones sdpa fast path takes.
    one_texts = [generate_batch(model, tokenizer, [prompts[i]], [images[i]],
                                do_sample=False, max_new_tokens=max_new,
                                seed=0)['texts'][0]
                 for i in range(len(prompts))]
    many = generate_batch(model, tokenizer, prompts, images, do_sample=False,
                          max_new_tokens=max_new, seed=0)

    def cmp(a, b, label):
        same = sum(1 for x, y in zip(a, b) if x == y)
        return same, len(a), f'{label} {same}/{len(a)} identical'

    s1, n1, d1 = cmp(one_texts, ref_texts, 'batch=1 vs inference()')
    s2, n2, d2 = cmp(many['texts'], ref_texts, f'batch={batch} vs inference()')
    gate('G2.3a equal-N batch equivalence', s1 == n1 and s2 == n2,
         f'{d1}; {d2}; N={eq[0]["n_visual"]}, max_new_tokens={max_new}')

    # negative control: same comparison with mixed visual-token counts
    mixed = mixed_n_pool[:batch]
    mprompts, mimages = prepare(mixed, store, tokenizer, len(mixed), model)
    mref = [inference(model, im, mixed[i]['prompt'], tokenizer, do_sample=False)
            for i, im in enumerate(mimages)]
    nm = len(mixed)
    note = 'diverged as predicted'
    try:
        mmany = generate_batch(model, tokenizer, mprompts, mimages,
                               do_sample=False, max_new_tokens=max_new, seed=0)
        sm, nm, _ = cmp(mmany['texts'], mref, 'mixed-N')
        note = f'{sm}/{nm} identical (divergence expected)'
        mixed_ok = sm < nm
    except ValueError as exc:            # the equal-N guard fired
        note = f'blocked by guard: {exc}'
        mixed_ok = True
        sm = 0
    ns = sorted({int(im.shape[0]) for im in mimages})
    gate('G2.3b mixed-N negative control', mixed_ok,
         f'N={ns}; {note} -> equal-N batching is load-bearing')
    return {'equal_n': {'batch1': s1, 'batchN': s2, 'n': n1},
            'mixed_n': {'identical': sm, 'n': nm, 'note': note}}


def _max_gap(logp_a, logp_b, lengths):
    diffs = []
    for k, lc in enumerate(lengths):
        diffs += (logp_a[k, :lc] - logp_b[k, :lc]).abs().tolist()
    return (max(diffs) if diffs else 0.0), len(diffs)


def _trim(gen):
    comp = gen['completion_ids']
    return [comp[b][gen['mask'][b]] for b in range(comp.shape[0])]


def g2_4_logprob_paths(model, tokenizer, entries, store, temperature, batch,
                       max_new):
    pad = tokenizer.pad_token_id
    prompts, images = prepare(entries, store, tokenizer, batch, model)

    # Unit temperature first: no warper runs on either side, so this isolates the
    # index arithmetic from dtype rounding. It is the real plumbing check.
    gen_unit = generate_batch(model, tokenizer, prompts, images, do_sample=True,
                              temperature=1.0, top_p=1.0, max_new_tokens=max_new,
                              seed=1234)
    rows_unit = _trim(gen_unit)
    b_unit = forward_sequence(model, prompts, rows_unit, images, temperature=1.0,
                              pad_id=pad)
    gap_u, ntok_u = _max_gap(gen_unit['logp'], b_unit['logp'],
                             [len(r) for r in rows_unit])

    # Tempered: path A's scores were divided by T *inside* the model dtype by
    # TemperatureLogitsWarper, path B divides in fp32, so a residual of bf16
    # rounding is expected here and is not an index error.
    gen_t = generate_batch(model, tokenizer, prompts, images, do_sample=True,
                           temperature=temperature, top_p=1.0,
                           max_new_tokens=max_new, seed=1234)
    rows_t = _trim(gen_t)
    b_t = forward_sequence(model, prompts, rows_t, images,
                           temperature=temperature, pad_id=pad)
    gap_t, ntok_t = _max_gap(gen_t['logp'], b_t['logp'], [len(r) for r in rows_t])

    c = forward_sequence(model, prompts, rows_unit, images, temperature=1.0,
                         with_labels=True, pad_id=pad)
    mean_b = -mean_logp(b_unit)
    mean_c = c['loss']
    rel = abs(mean_b - mean_c) / max(abs(mean_c), 1e-9)

    # The raw A-vs-B gap cannot be tiny in bf16, and is not load-bearing: generate
    # scores come from an incremental KV-cache decode while path B is one full
    # forward, and the two attention reductions round differently. diag_logprob_dtype
    # measured this gap shrinking 8.9x going bf16->fp16 (8->11 mantissa bits
    # predicts 2**3=8x), which identifies it as rounding. An index/off-by-one bug is
    # dtype-independent and would score the sampled tokens against the wrong
    # distribution, gapping by ~5+ nats toward log(vocab)~12; 0.5 sits far above the
    # ~0.08 bf16 rounding floor and far below any real mis-gather. The trainer never
    # uses path A's logp anyway -- old/ref/actor all come from path B (G2.4b/G2.4c).
    gate('G2.4a path A vs path B within bf16 rounding (T=1.0)', gap_u < 0.5,
         f'max |logp_A - logp_B| = {gap_u:.3e} over {ntok_u} tokens (bf16 '
         f'cache-vs-full rounding: fp16 shrinks it 8.9x per diag_logprob_dtype; '
         f'an index bug would gap ~5+ nats). Path B is the trainer\'s only logp '
         f'source -- proven exact by G2.4b, reproducible by G2.4c.')
    gate('G2.4b path B vs path C', rel < 1e-4,
         f'mean NLL_B = {mean_b:.6f} vs CE_loss_C = {mean_c:.6f} '
         f'(rel {rel:.2e}, {c["loss_tokens"]} tokens)')
    print(f'  [info] tempered gap at T={temperature}: {gap_t:.3e} over '
          f'{ntok_t} tokens -- bf16 rounding in TemperatureLogitsWarper, which '
          f'divides in the model dtype while path B divides in fp32. This is why '
          f'logp_old is recomputed with path B instead of taken from scores.',
          flush=True)
    return {'max_abs_diff_A_B_T1': gap_u, 'n_tokens_T1': ntok_u,
            'max_abs_diff_A_B_tempered': gap_t, 'n_tokens_tempered': ntok_t,
            'mean_nll_B': mean_b, 'loss_C': mean_c, 'rel_C': rel,
            'sample_texts': gen_t['texts'],
            'sample_tasks': [e['task'] for e in entries[:len(gen_t['texts'])]]}


def g2_4c_lora_neutrality(model, tokenizer, entries, store, batch, temperature,
                          max_new, bf16_texts):
    """ratio must be exactly 1 at step 0, or the first update is already clipped.

    GRPO divides exp(logp_actor - logp_old); both come from path B, so what has
    to hold is that path B is reproducible and that attaching a zero-initialised
    RL-LoRA leaves it bit-identical. A fresh LoRA's B matrix is zeros, so any
    nonzero gap here means the adapter is not neutral at init.
    """
    pad = tokenizer.pad_token_id
    prompts, images = prepare(entries, store, tokenizer, batch, model)
    rows = _trim(generate_batch(model, tokenizer, prompts, images, do_sample=True,
                                temperature=temperature, top_p=1.0,
                                max_new_tokens=max_new, seed=99))
    base = forward_sequence(model, prompts, rows, images, temperature=temperature,
                            pad_id=pad)
    again = forward_sequence(model, prompts, rows, images, temperature=temperature,
                             pad_id=pad)
    deterministic = torch.equal(base['logp'], again['logp'])

    lora_model = attach_rl_lora(model)
    n_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    wrapped = forward_sequence(lora_model, prompts, rows, images,
                               temperature=temperature, pad_id=pad)
    gap, ntok = _max_gap(base['logp'], wrapped['logp'], [len(r) for r in rows])

    greedy_texts = None
    if bf16_texts is not None:
        greedy_texts = generate_batch(lora_model, tokenizer, prompts, images,
                                      do_sample=False, max_new_tokens=max_new,
                                      seed=0)['texts']

    same_greedy = (greedy_texts is None
                   or greedy_texts == bf16_texts[:len(greedy_texts)])
    gate('G2.4c path B reproducible + RL-LoRA neutral at init',
         deterministic and gap == 0.0 and same_greedy,
         f'path B twice bit-identical={deterministic}; zero-init RL-LoRA '
         f'({n_trainable / 1e6:.1f}M trainable) max |logp| shift = {gap:.3e} '
         f'over {ntok} tokens; greedy texts unchanged={same_greedy}')
    return {'deterministic': bool(deterministic), 'lora_logp_gap': gap,
            'n_tokens': ntok, 'trainable_params': n_trainable,
            'greedy_texts_unchanged': bool(same_greedy)}


def g2_5_throughput(model, tokenizer, entries, store, sizes, temperature, max_new,
                    n_prompts, group_size):
    rows = []
    for B in sizes:
        pool = (entries * ((B // len(entries)) + 1))[:B]
        prompts, images = prepare(pool, store, tokenizer, B, model)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = generate_batch(model, tokenizer, prompts, images, do_sample=True,
                             temperature=temperature, top_p=1.0,
                             max_new_tokens=max_new, seed=7)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        gen_len = float(out['mask'].sum()) / B
        rows.append({'batch': B, 'seconds': round(dt, 2),
                     'items_per_s': round(B / dt, 2),
                     'peak_gib': round(peak, 2),
                     'mean_completion_tokens': round(gen_len, 1)})
        print(f'    B={B:2d}  {dt:6.2f}s  {B / dt:6.2f} items/s  '
              f'peak {peak:5.2f} GiB  mean L_c {gen_len:4.1f}', flush=True)

    per_item = next((r['items_per_s'] for r in rows if r['batch'] == 1), None)
    wide = [r for r in rows if r['batch'] >= 16] or [max(rows, key=lambda r: r['batch'])]
    best = max(r['items_per_s'] for r in wide)
    speedup = best / per_item if per_item else float('inf')
    rollouts = n_prompts * group_size
    hours = rollouts / best / 3600
    gate('G2.5 throughput', speedup >= 4.0 and hours <= 3.0,
         f'best {best:.1f} items/s at B={max(wide, key=lambda r: r["batch"])["batch"]} '
         f'= {speedup:.1f}x the per-item path (needs >=4x); one rollout epoch '
         f'({n_prompts} prompts x G={group_size} = {rollouts}) extrapolates to '
         f'{hours:.2f}h (M3 budget 3h)')
    return {'rows': rows, 'best_items_per_s': best, 'speedup_vs_batch1': speedup,
            'rollouts_per_epoch': rollouts, 'epoch_rollout_hours': round(hours, 2)}


def g2_6_dtype_fidelity(model_fp16, tokenizer, entries, store, batch, bf16_texts,
                        max_new):
    prompts, images = prepare(entries, store, tokenizer, batch, model_fp16)
    out = generate_batch(model_fp16, tokenizer, prompts, images, do_sample=False,
                         max_new_tokens=max_new, seed=0)
    n = min(len(out['texts']), len(bf16_texts))
    same = sum(1 for a, b in zip(out['texts'], bf16_texts) if a == b)
    gate('G2.6 bf16 vs fp16 greedy', same >= int(0.9 * n),
         f'{same}/{n} greedy completions identical between the dtype the '
         f'trainer uses (bf16) and the dtype the frozen eval uses (fp16)')
    return {'identical': same, 'n': n,
            'fp16_texts': out['texts'][:3], 'bf16_texts': bf16_texts[:3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2', default=DEFAULT_B3)
    ap.add_argument('--data', default=DEFAULT_TRAIN_JSON)
    ap.add_argument('--tasks', nargs='*', default=['time_grounding'])
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--sizes', type=int, nargs='*', default=[1, 4, 8, 16, 32])
    ap.add_argument('--group-size', type=int, default=8,
                    help='G per prompt, used to extrapolate one rollout epoch')
    ap.add_argument('--out', default='training_logs/smoke_m2.json')
    ap.add_argument('--skip-fp16', action='store_true')
    ap.add_argument('--skip-throughput', action='store_true')
    a = ap.parse_args()

    if a.data.endswith('.jsonl'):
        entries = read_jsonl(a.data)
        skipped = {}
    else:
        entries, skipped = load_entries(a.data, tasks=tuple(a.tasks) or None)
    print(f'[data] {len(entries)} entries, dropped {dict(skipped) or "none"}')
    if not entries:
        raise SystemExit('no entries to smoke-test')

    buckets = buckets_by_n_visual(entries)
    main_n = max(buckets, key=lambda k: len(buckets[k]))
    eq_pool = buckets[main_n]
    if a.batch > len(eq_pool):
        print(f'[data] WARNING --batch {a.batch} exceeds the main bucket '
              f'(N={main_n}, {len(eq_pool)} entries); clamping')
        a.batch = len(eq_pool)
    if a.batch < 1:
        raise SystemExit('main bucket is empty')

    distinct_ns = sorted(buckets)
    mixed_pool = [buckets[k][0] for k in distinct_ns][:a.batch]
    if len({e['n_visual'] for e in mixed_pool}) < 2:
        mixed_pool = entries[:a.batch]
    if len({e['n_visual'] for e in mixed_pool}) < 2:
        raise SystemExit('cannot build a mixed-N control: corpus has one N only')
    print(f'[data] main bucket N={main_n} ({len(eq_pool)} entries); '
          f'mixed control N={sorted({e["n_visual"] for e in mixed_pool})}')

    store = FeatureStore()
    print(f'[load] bf16 actor from {a.stage2}')
    tokenizer, model, ctx = load_merged_actor(a.stage2, dtype=torch.bfloat16)
    print(f'[load] done, context {ctx}, pad={tokenizer.pad_token_id}, '
          f'eos={tokenizer.eos_token_id}, padding_side={tokenizer.padding_side}')

    report = {'stage2': a.stage2, 'data': a.data, 'batch': a.batch,
              'temperature': a.temperature, 'dtype': 'bfloat16'}
    report['answer_lengths'] = g2_2_answer_lengths(eq_pool, tokenizer)
    # inference() allows 1024 new tokens; truncating shorter than the longest
    # real answer would make G2.3 report a divergence that is only an artifact.
    max_new = max(64, report['answer_lengths']['max'] + 8)
    report['max_new_tokens'] = max_new
    report['batch_equivalence'] = g2_3_batch_equivalence(
        model, tokenizer, eq_pool, store, a.batch, mixed_pool, max_new)
    report['logprob'] = g2_4_logprob_paths(
        model, tokenizer, eq_pool, store, a.temperature, min(4, a.batch), max_new)
    if not a.skip_throughput:
        report['throughput'] = g2_5_throughput(
            model, tokenizer, eq_pool, store, a.sizes, a.temperature, max_new,
            len(eq_pool), a.group_size)

    pf = parse_failures(report['logprob']['sample_tasks'],
                        report['logprob']['sample_texts'])
    report['sample_parse_failures'] = dict(pf)

    bf16_texts = None
    if not a.skip_fp16:
        prompts, images = prepare(eq_pool, store, tokenizer, a.batch, model)
        bf16_texts = generate_batch(model, tokenizer, prompts, images,
                                    do_sample=False, max_new_tokens=max_new,
                                    seed=0)['texts']

    # runs last on the bf16 actor: it leaves the model wrapped in a PEFT adapter
    report['lora_neutrality'] = g2_4c_lora_neutrality(
        model, tokenizer, eq_pool, store, min(4, a.batch), a.temperature,
        max_new, bf16_texts)

    if not a.skip_fp16:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print('[load] fp16 actor (same checkpoint) for the dtype fidelity gate')
        tokenizer16, model16, _ = load_merged_actor(a.stage2, dtype=torch.float16)
        report['dtype_fidelity'] = g2_6_dtype_fidelity(
            model16, tokenizer16, eq_pool, store, a.batch, bf16_texts, max_new)
        del model16
        gc.collect()
        torch.cuda.empty_cache()

    report['gates'] = RESULTS
    report['all_pass'] = all(r['pass'] for r in RESULTS)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    print(f'\n[M2] {sum(r["pass"] for r in RESULTS)}/{len(RESULTS)} gates passed'
          f' -> report {a.out}')
    for r in RESULTS:
        if not r['pass']:
            print(f'  FAILED {r["gate"]}: {r["detail"]}')
    print('  G2.3a/G2.4 是硬门：不过就不写 grpo_trainer。', flush=True)
    return 0 if report['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
