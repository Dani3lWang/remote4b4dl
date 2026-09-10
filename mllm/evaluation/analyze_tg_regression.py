#!/usr/bin/env python
"""TG (time_grounding) failure-mode comparison between two eval runs.

Reuses the exact interval regex + closed-interval IoU semantics of
evaluate_model.compute_miou so per-bucket numbers sum back to the
reported mIoU.  Usage:
    python analyze_tg_regression.py <runA_dir> <runB_dir> [--labels A B]
"""
import json
import re
import sys
from collections import Counter

import numpy as np

PAT = re.compile(
    r'from\s+frame\s+(\d+)\s+to\s+frame\s+(\d+)'
    r'|frame\s+(\d+)\s+(?:to|-|and)\s+frame\s+(\d+)'
    r'|from\s+frame\s+(\d+)\s+until\s+frame\s+(\d+)')


def parse(text: str):
    if not text:
        return None
    m = PAT.search(text.lower())
    if not m:
        return None
    nums = [int(x) for x in m.groups() if x is not None]
    return (nums[0], nums[1])


def load_tg(path):
    d = json.load(open(path))
    tg = d['time_grounding']
    return tg['predictions'], tg['ground_truths'], tg['questions']


def iou(pred, gt):
    (ps, pe), (gs, ge) = pred, gt
    inter = min(pe, ge) - max(ps, gs) + 1
    if inter <= 0:
        return 0.0
    union = (pe - ps + 1) + (ge - gs + 1) - inter
    return inter / union


def analyze(run_dir, label):
    preds_raw, gts_raw, qs = load_tg(f'{run_dir}/predictions.json')
    preds = [parse(p) for p in preds_raw]
    gts = [parse(g) for g in gts_raw]
    n = len(gts)
    ious = np.array([iou(p, g) if (p and g) else 0.0
                     for p, g in zip(preds, gts)])
    valid = np.array([p is not None for p in preds])

    print(f'\n================ {label} ({run_dir}) ================')
    print(f'samples={n}  parse-fail(pred)={n-int(valid.sum())} ({1-valid.mean():.1%})  '
          f'parse-fail(gt)={sum(1 for g in gts if g is None)}')
    print(f'mIoU(closed-int) = {ious.mean():.4f}')

    hit_exact = sum(1 for p, g in zip(preds, gts) if p and p == g)
    hit_05 = (ious >= 0.5).sum()
    zero = (ious == 0).sum()
    print(f'exact match {hit_exact} ({hit_exact/n:.1%}) | IoU>=0.5 {hit_05} '
          f'({hit_05/n:.1%}) | IoU=0 {zero} ({zero/n:.1%})')

    # ---- offsets: pred - gt on starts / ends / center ----
    pairs = [(p, g) for p, g in zip(preds, gts) if p and g]
    ds = np.array([p[0]-g[0] for p, g in pairs])
    de = np.array([p[1]-g[1] for p, g in pairs])
    dc = np.array([(p[0]+p[1])/2-(g[0]+g[1])/2 for p, g in pairs])
    dl = np.array([(p[1]-p[0])-(g[1]-g[0]) for p, g in pairs])
    print(f'offset (pred-gt, only parseable pairs n={len(pairs)}): '
          f'start mean {ds.mean():+.2f} med {np.median(ds):+.1f} | '
          f'end mean {de.mean():+.2f} | center mean {dc.mean():+.2f} med {np.median(dc):+.1f}')
    print(f'pred early(center<-2) {(dc<-2).mean():.1%}  late(center>2) {(dc>2).mean():.1%}')
    print(f'interval length: GT {np.mean([g[1]-g[0]+1 for _,g in pairs]):.1f} '
          f'vs pred {np.mean([p[1]-p[0]+1 for p,_ in pairs]):.1f} frames')

    # ---- buckets by GT start ----
    bounds = [0, 10, 20, 25, 30, 35, 40]
    gt_start = np.array([g[0] for g in gts if g])
    valid_idx = [i for i, g in enumerate(gts) if g]
    print('\nbucket(GT start) | GT share | pred-start share | n | mIoU | IoU>=0.5 | center-offset')
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = [(gt_start >= a) & (gt_start < b)][0]
        idx = [i for i in valid_idx if sel[i]]
        if not idx:
            continue
        gt_share = len(idx) / n
        pred_share = sum(1 for i in idx if preds[i] and a <= preds[i][0] < b) / n
        m = ious[idx].mean()
        h05 = (ious[idx] >= 0.5).sum() / len(idx)
        off = np.mean([(preds[i][0]+preds[i][1])/2-(gts[i][0]+gts[i][1])/2
                       for i in idx if preds[i]])
        print(f'[{a:2d},{b:2d})        | {gt_share:6.1%}    | {pred_share:6.1%}'
              f'        | {len(idx):4d} | {m:.4f} | {h05:5.1%}   | {off:+.1f}')

    # ---- where do predictions land when they miss (GT start>=25)? ----
    hi = [i for i in valid_idx if gt_start[i] >= 25]
    if hi:
        ph = Counter()
        for i in hi:
            p = preds[i]
            ph[None if not p else '0-9' if p[0] < 10 else '10-19' if p[0] < 20
               else '20-24' if p[0] < 25 else '25-29' if p[0] < 30
               else '30-34' if p[0] < 35 else '35+'] += 1
        print(f'\nGT start>=25 (n={len(hi)}): predicted-start distribution '
              f'{dict(sorted(ph.items(), key=lambda kv: (kv[0] is None, kv[0])))}')

    # ---- most common predicted intervals (collapse check) ----
    top = Counter(p for p in preds if p).most_common(5)
    print('top-5 predicted intervals: ' +
          ', '.join(f'[{a:02d}-{b:02d}] x{c} ({c/n:.1%})' for (a, b), c in top))
    return ious, preds, gts


def main():
    run_a, run_b = sys.argv[1], sys.argv[2]
    lab_a = sys.argv[4] if len(sys.argv) > 4 else 'A'
    lab_b = sys.argv[5] if len(sys.argv) > 5 else 'B'
    ia, pa, ga = analyze(run_a, lab_a)
    ib, pb, gb = analyze(run_b, lab_b)
    n = len(ia)
    print(f'\n==== per-sample IoU delta ({lab_b} - {lab_a}, n={n}) ====')
    dd = ib - ia
    print(f'delta mIoU {dd.mean():+.4f}')
    hi = [i for i in range(n) if ga[i] and ga[i][0] >= 25]
    lo = [i for i in range(n) if ga[i] and ga[i][0] < 25]
    print(f'GT start>=25: {dd[hi].mean():+.4f} (n={len(hi)}) | '
          f'GT start<25: {dd[lo].mean():+.4f} (n={len(lo)})')
    # who improved / regressed among parseable
    reg = sum(1 for i in range(n) if dd[i] < -0.05)
    imp = sum(1 for i in range(n) if dd[i] > 0.05)
    print(f'regressed >0.05 IoU: {reg} ({reg/n:.1%}) | improved >0.05: {imp} ({imp/n:.1%})')


if __name__ == '__main__':
    main()
