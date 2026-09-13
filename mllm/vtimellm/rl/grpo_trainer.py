"""GRPO trainer for B4DL time-grounding -- one process, one 32GB GPU.

Why one process when the plan (§5) sketched three: the reference policy is the
merged B3 SFT base with the RL-LoRA *disabled* (`model.disable_adapter()`), not a
second resident copy, so rollout / reference / actor all share one ~14GB model.
Only the ~160M RL-LoRA parameters train (the merged base is frozen), so ZeRO-3
and CPU offload are unnecessary; peak memory is held down by micro-batching the
forward/backward instead. `smoke_m2` measured the merged actor at 12.7-16.0 GiB
across batch 1-32, which leaves room for the LoRA optimizer states.

logp_old, logp_ref and logp_actor are ALL recomputed with path B
(`logprob.forward_sequence`); the rollout's path-A scores are used only to pick
tokens, never as log-probabilities. See smoke_m2 G2.4 (path B == path C, the
model's own CE, to 1e-7) and diag_logprob_dtype for why.

Precision: LoRA weights live in bf16 so every forward (including `generate`)
runs natively, but a fp32 master copy is what AdamW updates, then casts back to
bf16 each step. At lr=1e-5 a pure-bf16 update is below the bf16 ULP of a ~1e-2
weight and would silently round to zero; the fp32 master accumulates those
sub-ULP steps. This is the standard mixed-precision master-weight pattern.
"""

import argparse
import json
import math
import os
import time
from collections import Counter
from contextlib import nullcontext

import torch

from . import DEFAULT_B3, DEFAULT_TRAIN_JSON
from .data import (FeatureStore, buckets_by_n_visual, iter_prompt_batches,
                   load_entries, read_jsonl)
from .logprob import forward_sequence
from .model_utils import attach_rl_lora, encode_prompt, load_merged_actor
from .reward import EV, group_advantage, parse_failures, score_batch
from .rollout import generate_batch


def slices(n, m):
    for i in range(0, n, m):
        yield list(range(i, min(i + m, n)))


def make_lr_lambda(warmup, total):
    def f(step):
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return f


def trim_completions(gen):
    comp = gen['completion_ids']
    return [comp[b][gen['mask'][b]] for b in range(comp.shape[0])]


def pad_rows_to(per_row, L, device):
    out = torch.zeros((len(per_row), L), dtype=torch.float32, device=device)
    for k, v in enumerate(per_row):
        lc = min(len(v), L)
        out[k, :lc] = v[:lc].to(device)
    return out


def collect_logp(model, prompts, completions, images, pad, temperature, micro,
                 disable_adapter=False):
    """Per-row completion logp (list of 1-D fp32 CPU tensors), path B, no grad.

    disable_adapter=True scores the frozen SFT base (the GRPO reference); False
    scores the current actor. Variable per-row lengths let any later micro-batch
    re-pad its own slice to its own width.
    """
    per_row = [None] * len(prompts)
    ctx = model.disable_adapter() if disable_adapter else nullcontext()
    with torch.no_grad(), ctx:
        for sl in slices(len(prompts), micro):
            fwd = forward_sequence(
                model, [prompts[i] for i in sl], [completions[i] for i in sl],
                [images[i] for i in sl], temperature=temperature, pad_id=pad,
                to_cpu=True)
            for k, i in enumerate(sl):
                lc = fwd['completion_lengths'][k]
                per_row[i] = fwd['logp'][k, :lc].clone()
    return per_row


def center_stats(tasks, texts):
    """TG collapse triad: predicted-centre std, mode share and entropy.

    A shrinking std / rising mode share / falling entropy is the policy
    collapsing onto one frame guess (reward hacking), the failure mode the plan
    §1.5 warns about. Uses EV.extract_time_grounding_frames, never a re-impl.
    """
    centers = []
    for t, x in zip(tasks, texts):
        if t != 'time_grounding':
            continue
        d = EV.extract_time_grounding_frames(x or '')
        if d and d.get('start_frame') is not None and d.get('end_frame') is not None:
            centers.append((d['start_frame'] + d['end_frame']) / 2.0)
    if not centers:
        return {}
    n = len(centers)
    mean = sum(centers) / n
    std = (sum((c - mean) ** 2 for c in centers) / n) ** 0.5
    rounded = [round(c) for c in centers]
    cnt = Counter(rounded)
    mode_share = max(cnt.values()) / n
    entropy = -sum((v / n) * math.log(v / n) for v in cnt.values())
    return {'center_mean': round(mean, 3), 'center_std': round(std, 3),
            'center_mode_share': round(mode_share, 4),
            'center_entropy': round(entropy, 4), 'n_centers': n}


def train_one_step(model, master, optimizer, named_trainable, prompts,
                   completions, images, pad, temperature, adv_rows, keep_rows,
                   logp_old, logp_ref, micro, eps, beta, grad_clip, lr):
    """Micro-batched GRPO update; returns stats or None if every group degenerate."""
    kept_total = sum(1 for k in keep_rows if k)
    if kept_total == 0:
        return None
    device = next(model.parameters()).device
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    for _, p in named_trainable:
        p.grad = None

    model.train()
    loss_sum = kl_sum = cf_num = cf_den = ratio_dev_sum = 0.0
    for sl in slices(len(prompts), micro):
        fwd = forward_sequence(
            model, [prompts[i] for i in sl], [completions[i] for i in sl],
            [images[i] for i in sl], temperature=temperature, pad_id=pad,
            to_cpu=False)
        logp_a = fwd['logp']                          # (mb, L) fp32, grad, on device
        msk = fwd['mask'].to(logp_a.device).float()
        L = logp_a.shape[1]
        old = pad_rows_to([logp_old[i] for i in sl], L, logp_a.device)
        ref = pad_rows_to([logp_ref[i] for i in sl], L, logp_a.device)
        adv_t = torch.tensor([adv_rows[i] for i in sl], device=logp_a.device,
                             dtype=torch.float32).unsqueeze(1)
        keep_t = torch.tensor([1.0 if keep_rows[i] else 0.0 for i in sl],
                              device=logp_a.device, dtype=torch.float32).unsqueeze(1)

        ratio = torch.exp(logp_a - old)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv_t
        pg_loss = -torch.min(surr1, surr2)
        kl = torch.exp(ref - logp_a) - (ref - logp_a) - 1.0   # k3, non-negative
        tok = (pg_loss + beta * kl) * msk
        per_seq = tok.sum(1) / msk.sum(1).clamp(min=1.0)      # token-mean in seq
        mb_loss = (per_seq * keep_t.squeeze(1)).sum() / kept_total
        mb_loss.backward()
        loss_sum += float(mb_loss.detach())
        with torch.no_grad():
            km = msk * keep_t
            kl_sum += float((kl * km).sum())
            ratio_dev_sum += float(((ratio - 1.0).abs() * km).sum())
            cf = (((ratio < 1.0 - eps) | (ratio > 1.0 + eps)).float()) * km
            cf_num += float(cf.sum())
            cf_den += float(km.sum())

    # bf16 grads -> fp32 master, step, cast master back to bf16.
    for n, p in named_trainable:
        master[n].grad = (p.grad.detach().float() if p.grad is not None
                          else torch.zeros_like(master[n]))
    grad_norm = torch.nn.utils.clip_grad_norm_(list(master.values()), grad_clip)
    optimizer.step()
    for n, p in named_trainable:
        p.data.copy_(master[n].to(p.dtype))
        p.grad = None

    return {'loss': round(loss_sum, 5),
            'kl_mean': round(kl_sum / max(cf_den, 1.0), 5),
            'clip_frac': round(cf_num / max(cf_den, 1.0), 5),
            'ratio_dev': round(ratio_dev_sum / max(cf_den, 1.0), 6),
            'grad_norm': round(float(grad_norm), 4), 'lr': lr}


def latest_ckpt(out_dir):
    if not os.path.isdir(out_dir):
        return None, 0
    best, step = None, 0
    for d in os.listdir(out_dir):
        if d.startswith('rl_lora_step'):
            try:
                s = int(d.split('step')[-1])
            except ValueError:
                continue
            if s > step:
                step, best = s, os.path.join(out_dir, d)
    return best, step


def prune(out_dir, keep):
    dirs = sorted((os.path.join(out_dir, d) for d in os.listdir(out_dir)
                   if d.startswith('rl_lora_step')),
                  key=lambda p: int(p.split('step')[-1]))
    for d in dirs[:-keep] if keep > 0 else []:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        os.rmdir(d)
        print(f'  [prune] removed superseded {d}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2', default=DEFAULT_B3)
    ap.add_argument('--data', default=DEFAULT_TRAIN_JSON)
    ap.add_argument('--tasks', nargs='*', default=['time_grounding'])
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--prompts-per-step', type=int, default=32)
    ap.add_argument('--group-size', type=int, default=8)
    ap.add_argument('--micro-batch', type=int, default=4)
    ap.add_argument('--gen-max-batch', type=int, default=32)
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--max-new', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--beta', type=float, default=0.04)
    ap.add_argument('--eps', type=float, default=0.2)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--warmup-ratio', type=float, default=0.03)
    ap.add_argument('--inner-epochs', type=int, default=1)
    ap.add_argument('--save-steps', type=int, default=50)
    ap.add_argument('--save-total-limit', type=int, default=2)
    ap.add_argument('--log-steps', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()

    P, G = a.prompts_per_step, a.group_size
    torch.manual_seed(a.seed)

    if a.data.endswith('.jsonl'):
        entries = read_jsonl(a.data)
    else:
        entries, _ = load_entries(a.data, tasks=tuple(a.tasks) or None)
    buckets = buckets_by_n_visual(entries)
    steps_per_epoch = sum(len(v) // P for v in buckets.values())
    if steps_per_epoch == 0:
        raise SystemExit(f'no full batch of P={P}; corpus too small per N-bucket')
    total_steps = a.max_steps if a.max_steps else int(math.ceil(a.epochs * steps_per_epoch))
    warmup = int(a.warmup_ratio * total_steps)
    lr_lambda = make_lr_lambda(warmup, total_steps)
    print(f'[data] {len(entries)} entries, N-buckets {sorted(buckets)}; '
          f'P={P} G={G} -> {steps_per_epoch} steps/epoch, total {total_steps}, '
          f'warmup {warmup}', flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    store = FeatureStore()
    print(f'[load] actor + RL-LoRA from {a.stage2}', flush=True)
    tokenizer, model, _ = load_merged_actor(a.stage2, dtype=torch.bfloat16)
    model = attach_rl_lora(model)
    pad = tokenizer.pad_token_id
    actor_dtype = torch.bfloat16

    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for _, p in named_trainable)
    master = {n: p.detach().clone().float() for n, p in named_trainable}
    for v in master.values():
        v.requires_grad_(False)
    optimizer = torch.optim.AdamW(list(master.values()), lr=a.lr, weight_decay=0.0)
    print(f'[load] RL-LoRA trainable {n_train/1e6:.1f}M across '
          f'{len(named_trainable)} tensors; fp32 master + AdamW(wd=0)', flush=True)

    start_step = 0
    if a.resume:
        ck, s = latest_ckpt(a.out_dir)
        if ck:
            from peft import set_peft_model_state_dict
            import safetensors.torch
            sd = safetensors.torch.load_file(os.path.join(ck, 'adapter_model.safetensors'))
            set_peft_model_state_dict(model, sd)
            # rebuild the fp32 master from the resumed bf16 weights; Adam moments
            # restart from zero (they are not checkpointed -- a minor approximation
            # that costs a few steps of momentum after a resume, not correctness).
            master = {n: p.detach().clone().float() for n, p in named_trainable}
            optimizer = torch.optim.AdamW(list(master.values()), lr=a.lr, weight_decay=0.0)
            start_step = s
            print(f'[resume] loaded {ck} at step {s}; fp32 master rebuilt, '
                  f'optimizer moments restarted', flush=True)

    master0 = {n: v.clone() for n, v in master.items()}   # step-0 ref for lora_delta
    log_path = os.path.join(a.out_dir, 'grpo_log.jsonl')
    log_fh = open(log_path, 'a', encoding='utf-8')
    global_step = start_step
    t_start = time.time()
    done = False
    epoch = start_step // steps_per_epoch
    skip = start_step % steps_per_epoch     # only the first (resumed) epoch fast-forwards
    while not done and global_step < total_steps:
        batch_iter = iter_prompt_batches(entries, P, repeat=G, shuffle=True,
                                         seed=a.seed + epoch)
        for _ in range(skip):               # CPU only, no GPU work
            try:
                next(batch_iter)
            except StopIteration:
                break
        skip = 0
        for batch in batch_iter:
            if global_step >= total_steps:
                done = True
                break
            step_t0 = time.time()
            prompts, images, tasks, gts = [], [], [], []
            for e in batch:
                _, ids, _ = encode_prompt(tokenizer, e['prompt'])
                prompts.append(ids)
                images.append(store.get(e).to(actor_dtype))
                tasks.append(e['task'])
                gts.append(e['gt'])

            model.eval()                    # the mode G2.3a validated generate in
            # Chunk the rollout: a single generate for all P*G rows builds a KV
            # cache for the whole group (128 rows ~13G, 256 ~26G) on top of the
            # resident model + AdamW state and OOMs the 32G card (seen at M3.2
            # step 1, when the optimizer state allocated at step 0 first landed).
            # Equal-N already holds within the step, so sub-batches stay valid;
            # smoke G2.5 measured B=32 at 38.7 items/s, near peak throughput.
            torch.manual_seed(a.seed * 100003 + global_step)
            completions, texts = [], []
            for sl in slices(len(prompts), a.gen_max_batch):
                gen = generate_batch(model, tokenizer,
                                     [prompts[i] for i in sl],
                                     [images[i] for i in sl], do_sample=True,
                                     temperature=a.temperature, top_p=1.0,
                                     max_new_tokens=a.max_new, seed=None)
                completions.extend(trim_completions(gen))
                texts.extend(gen['texts'])
            torch.cuda.empty_cache()
            rewards = score_batch(tasks, texts, gts)
            adv, keep, degenerate, n_groups = group_advantage(rewards, G)
            adv_rows = [adv[p][g] for p in range(n_groups) for g in range(G)]
            keep_rows = [keep[p] for p in range(n_groups) for g in range(G)]

            # dropout=0 and no batchnorm make eval/train numerically identical
            # here, but old/ref/actor are all scored in train mode so the step-0
            # ratio exp(logp_actor - logp_old) is exactly 1 (G2.4c).
            model.train()
            logp_old = collect_logp(model, prompts, completions, images, pad,
                                    a.temperature, a.micro_batch, disable_adapter=False)
            logp_ref = collect_logp(model, prompts, completions, images, pad,
                                    a.temperature, a.micro_batch, disable_adapter=True)

            peak_before = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            stats = None
            for _ in range(a.inner_epochs):
                stats = train_one_step(
                    model, master, optimizer, named_trainable, prompts,
                    completions, images, pad, a.temperature, adv_rows, keep_rows,
                    logp_old, logp_ref, a.micro_batch, a.eps, a.beta,
                    a.grad_clip, a.lr * lr_lambda(global_step))
            peak = torch.cuda.max_memory_allocated() / 2 ** 30

            r_mean = sum(rewards) / len(rewards)
            r_var = sum((x - r_mean) ** 2 for x in rewards) / len(rewards)
            pf = parse_failures(tasks, texts)
            rec = {'step': global_step, 'epoch': epoch,
                   'reward_mean': round(r_mean, 4), 'reward_std': round(r_var ** 0.5, 4),
                   'degenerate_groups': degenerate, 'n_groups': n_groups,
                   'degenerate_rate': round(degenerate / max(n_groups, 1), 4),
                   'parse_unparsed': pf.get('unparsed', 0), 'parse_n': pf.get('n', 0),
                   'peak_gib': round(peak, 2),
                   'rollout_s': round(time.time() - step_t0, 2)}
            rec.update(center_stats(tasks, texts))
            if stats:
                if global_step == start_step:   # did the fp32-master update move weights?
                    stats['lora_delta'] = round(max(
                        (float((master[n] - master0[n]).abs().max()) for n in master),
                        default=0.0), 8)
                rec.update(stats)
            else:
                rec['skipped'] = 'all groups degenerate'
            log_fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            log_fh.flush()
            if global_step % a.log_steps == 0:
                print(f'  [step {global_step}] r={rec["reward_mean"]} '
                      f'std={rec["reward_std"]} degen={degenerate}/{n_groups} '
                      f'loss={rec.get("loss")} kl={rec.get("kl_mean")} '
                      f'rdev={rec.get("ratio_dev")} gnorm={rec.get("grad_norm")} '
                      f'cstd={rec.get("center_std")} peak={peak:.1f}G '
                      f'{rec["rollout_s"]}s', flush=True)

            global_step += 1
            if a.save_steps and global_step % a.save_steps == 0 and global_step < total_steps:
                d = os.path.join(a.out_dir, f'rl_lora_step{global_step}')
                model.save_pretrained(d)
                with open(os.path.join(a.out_dir, 'trainer_state.json'), 'w') as fh:
                    json.dump({'global_step': global_step, 'epoch': epoch}, fh)
                print(f'  [save] {d}', flush=True)
                prune(a.out_dir, a.save_total_limit)
        epoch += 1

    final = os.path.join(a.out_dir, f'rl_lora_step{global_step}')
    model.save_pretrained(final)
    with open(os.path.join(a.out_dir, 'trainer_state.json'), 'w') as fh:
        json.dump({'global_step': global_step, 'epoch': epoch,
                   'total_steps': total_steps}, fh)
    prune(a.out_dir, a.save_total_limit)
    log_fh.close()
    print(f'\n[GRPO] {global_step} steps in {(time.time()-t_start)/60:.1f} min '
          f'-> {final}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
