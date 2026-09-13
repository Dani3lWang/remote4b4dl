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
from .model_utils import encode_prompt, load_merged_actor
from .reward import parse_failures
from .rollout import generate_batch

RESULTS = []


def gate(name, passed, detail):
    RESULTS.append({'gate': name, 'pass': bool(passed), 'detail': detail})
    print(f'  [{"PASS" if passed else "FAIL"}] {name}: {detail}', flush=True)


def prepare(entries, store, tokenizer, n_items):
    """Prompt ids + feature tensors for the first n_items entries."""
    prompts, images = [], []
    for e in entries[:n_items]:
        _, ids, _ = encode_prompt(tokenizer, e['prompt'])
        prompts.append(ids)
        images.append(store.get(e))
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
    prompts, images = prepare(eq, store, tokenizer, batch)

    ref_texts = [inference(model, im, eq[i]['prompt'], tokenizer,
                           do_sample=False, frame_indices=None)
                 for i, im in enumerate(images)]
    one = generate_batch(model, tokenizer, prompts, images, do_sample=False,
                         max_new_tokens=max_new, seed=0)
    many = generate_batch(model, tokenizer, prompts, images, do_sample=False,
                          max_new_tokens=max_new, seed=0)

    def cmp(a, b, label):
        same = sum(1 for x, y in zip(a, b) if x == y)
        return same, len(a), f'{label} {same}/{len(a)} identical'

    s1, n1, d1 = cmp(one['texts'], ref_texts, 'batch=1 vs inference()')
    s2, n2, d2 = cmp(many['texts'], ref_texts, f'batch={batch} vs inference()')
    gate('G2.3a equal-N batch equivalence', s1 == n1 and s2 == n2,
         f'{d1}; {d2}; N={eq[0]["n_visual"]}, max_new_tokens={max_new}')

    # negative control: same comparison with mixed visual-token counts
    mixed = mixed_n_pool[:batch]
    mprompts, mimages = prepare(mixed, store, tokenizer, len(mixed))
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


def g2_4_logprob_paths(model, tokenizer, entries, store, temperature, batch,
                       max_new):
    prompts, images = prepare(entries, store, tokenizer, batch)
    gen = generate_batch(model, tokenizer, prompts, images, do_sample=True,
                         temperature=temperature, top_p=1.0,
                         max_new_tokens=max_new, seed=1234)
    comp = gen['completion_ids']
    rows = [comp[b][gen['mask'][b]] for b in range(comp.shape[0])]

    b_tempered = forward_sequence(model, prompts, rows, images,
                                  temperature=temperature,
                                  pad_id=tokenizer.pad_token_id)
    a = gen['logp']
    b = b_tempered['logp']
    diffs = []
    for k in range(len(rows)):
        lc = len(rows[k])
        diffs += (a[k, :lc] - b[k, :lc]).abs().tolist()
    max_diff = max(diffs) if diffs else 0.0

    c = forward_sequence(model, prompts, rows, images, temperature=1.0,
                         with_labels=True, pad_id=tokenizer.pad_token_id)
    b_unit = forward_sequence(model, prompts, rows, images, temperature=1.0,
                              pad_id=tokenizer.pad_token_id)
    mean_b = -mean_logp(b_unit)
    mean_c = c['loss']
    rel = abs(mean_b - mean_c) / max(abs(mean_c), 1e-9)

    gate('G2.4a path A vs path B', max_diff < 1e-3,
         f'max |logp_A - logp_B| = {max_diff:.3e} over {len(diffs)} tokens '
         f'(T={temperature})')
    gate('G2.4b path B vs path C', rel < 1e-4,
         f'mean NLL_B = {mean_b:.6f} vs CE_loss_C = {mean_c:.6f} '
         f'(rel {rel:.2e}, {c["loss_tokens"]} tokens)')
    return {'max_abs_diff_A_B': max_diff, 'n_tokens': len(diffs),
            'mean_nll_B': mean_b, 'loss_C': mean_c, 'rel_C': rel,
            'sample_texts': gen['texts'],
            'sample_tasks': [e['task'] for e in entries[:len(gen['texts'])]]}


def g2_5_throughput(model, tokenizer, entries, store, sizes, temperature, max_new):
    rows = []
    for B in sizes:
        pool = (entries * ((B // len(entries)) + 1))[:B]
        prompts, images = prepare(pool, store, tokenizer, B)
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
    wide = [r['items_per_s'] for r in rows if r['batch'] >= 8]
    best = max(wide) if wide else 0.0
    gate('G2.5 throughput', best >= 40.0,
         f'batch>=8 reaches {best:.1f} items/s (plan needs >=40; batch=1 eval '
         f'baseline was 17.5)' + ('' if wide else f'; no size >=8 in {sizes}'))
    return rows


def g2_6_dtype_fidelity(model_fp16, tokenizer, entries, store, batch, bf16_texts,
                        max_new):
    prompts, images = prepare(entries, store, tokenizer, batch)
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
    ap.add_argument('--sizes', type=int, nargs='*', default=[1, 4, 8, 16])
    ap.add_argument('--out', default='training_logs/smoke_m2.json')
    ap.add_argument('--skip-fp16', action='store_true')
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
    report['throughput'] = g2_5_throughput(
        model, tokenizer, eq_pool, store, a.sizes, a.temperature, max_new)

    pf = parse_failures(report['logprob']['sample_tasks'],
                        report['logprob']['sample_texts'])
    report['sample_parse_failures'] = dict(pf)

    bf16_texts = None
    if not a.skip_fp16:
        prompts, images = prepare(eq_pool, store, tokenizer, a.batch)
        bf16_texts = generate_batch(model, tokenizer, prompts, images,
                                    do_sample=False, max_new_tokens=max_new,
                                    seed=0)['texts']
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
