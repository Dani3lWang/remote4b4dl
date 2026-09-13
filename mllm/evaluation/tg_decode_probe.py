#!/usr/bin/env python
"""Zero-cost decode-side probe for time_grounding: is the frame-position signal
present in the model's output but suppressed by an answer prior?

Uses the exact interval regex + closed-interval IoU of evaluate_model.compute_miou,
so the "raw" row reproduces the reported mIoU and every delta is on the same scale.

IMPORTANT on interpretation: every fit below except --prior is tuned on the eval
ground truth itself, so those rows are upper bounds on what any decode-time
correction could buy, not reportable scores.  The diagnostic that actually decides
the route is the signal test (rank correlation + OLS slope + mode collapse).

Usage:
    python tg_decode_probe.py <runA> <runB> [--labels A B] [--prior train_centers.json]

<runA>/<runB> are eval result dirs (containing predictions.json) or .json files.
--prior is a JSON array of GT frame indices from the *training* set, used for the
one non-oracle correction (quantile map onto the train prior).
"""
import argparse
import json
import math
import os
import re
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

PAT = re.compile(
    r'from\s+frame\s+(\d+)\s+to\s+frame\s+(\d+)'
    r'|frame\s+(\d+)\s+(?:to|-|and)\s+frame\s+(\d+)'
    r'|from\s+frame\s+(\d+)\s+until\s+frame\s+(\d+)')


def parse(text):
    if not text:
        return None
    m = PAT.search(text.lower())
    if not m:
        return None
    nums = [int(x) for x in m.groups() if x is not None]
    return (nums[0], nums[1])


def load_any(p):
    path = p if p.endswith('.json') else os.path.join(p, 'predictions.json')
    with open(path, encoding='utf-8') as fh:
        tg = json.load(fh)['time_grounding']
    return ([parse(x) for x in tg['predictions']],
            [parse(x) for x in tg['ground_truths']],
            tg['questions'], path)


def iou(p, g):
    (ps, pe), (gs, ge) = p, g
    if ps > pe:
        ps, pe = pe, ps
    inter = min(pe, ge) - max(ps, gs) + 1
    if inter <= 0:
        return 0.0
    return inter / ((pe - ps + 1) + (ge - gs + 1) - inter)


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in order[i:j + 1]:
            r[k] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(xs, ys):
    if len(xs) < 3:
        return float('nan')
    mx, my = mean(xs), mean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float('nan')


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def ols(xs, ys):
    n = len(xs)
    if n < 3:
        return float('nan'), float('nan'), float('nan')
    mx, my = mean(xs), mean(ys)
    sxx = sum((a - mx) ** 2 for a in xs)
    if sxx == 0:
        return 0.0, my, 0.0
    slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sxx
    return slope, my - slope * mx, pearson(xs, ys) ** 2


def quantile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    pos = min(max(q, 0.0), 1.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def midrank_pct(sorted_vals, x):
    n = len(sorted_vals)
    lo, hi = bisect_left(sorted_vals, x), bisect_right(sorted_vals, x)
    return ((lo + hi) / 2 - 0.5) / n


def rebuild(center, length, hi_clip):
    length = max(1, int(round(length)))
    s = int(round(center - (length - 1) / 2.0))
    s = max(0, min(s, hi_clip - length + 1))
    return (s, s + length - 1)


def probe(preds, gts, qs, label, prior=None):
    n = len(gts)
    ok = [i for i in range(n) if preds[i] and gts[i]]
    bad = sum(1 for i in range(n) if not preds[i])
    raw = mean([iou(preds[i], gts[i]) if preds[i] and gts[i] else 0.0
                for i in range(n)])
    pc = [(preds[i][0] + preds[i][1]) / 2.0 for i in ok]
    pl = [preds[i][1] - preds[i][0] + 1 for i in ok]
    gc = [(gts[i][0] + gts[i][1]) / 2.0 for i in ok]
    gl = [gts[i][1] - gts[i][0] + 1 for i in ok]
    hi_clip = max(max(g[1] for g in gts if g), 1)
    ref_c = sorted(gc)
    ref_l = sorted(gl)

    print(f'\n================ {label} ================')
    print(f'n={n}  parse_fail={bad}  mIoU_raw={raw:.4f}')

    # ---- 1. signal test: is the output informative, or a collapsed prior? ----
    cc, ic, r2 = ols(pc, gc)
    cl, il, r2l = ols(pl, gl)
    top_c = Counter(pc).most_common(1)[0]
    top_i = Counter(tuple(preds[i]) for i in ok).most_common(1)[0]
    ent = -sum((c / len(pc)) * math.log2(c / len(pc)) for _, c in
               Counter(int(round(x)) for x in pc).items())
    print(f'[signal] center: spearman={spearman(pc, gc):+.3f} pearson={pearson(pc, gc):+.3f} '
          f'OLS slope={cc:+.3f} intercept={ic:+.2f} R2={r2:.3f}')
    print(f'         spread: pred std={std(pc):.2f} vs GT std={std(gc):.2f} '
          f'(ratio {std(pc)/std(gc):.2f}) | pred mean {mean(pc):.2f} vs GT mean {mean(gc):.2f}')
    print(f'         collapse: distinct centers={len(set(pc))} top-1 center '
          f'{top_c[0]:.1f} x{top_c[1]} ({top_c[1]/n:.1%}) | top-1 interval '
          f'{top_i[0]} x{top_i[1]} ({top_i[1]/n:.1%}) | center entropy={ent:.2f} bit '
          f'({math.log2(len(ref_c)):.2f} max)')
    print(f'         length: spearman={spearman(pl, gl):+.3f} slope={cl:+.3f} R2={r2l:.3f} '
          f'| pred mean={mean(pl):.1f} vs GT mean={mean(gl):.1f}')
    qgroups = defaultdict(list)
    for j, i in enumerate(ok):
        qgroups[qs[i]].append(gc[j])
    multi = {k: v for k, v in qgroups.items() if len(v) >= 5}
    if multi:
        within = mean([std(v) for v in multi.values()])
        print(f'         question-conditioned: {len(multi)} questions with >=5 samples, '
              f'within-question GT-center std={within:.2f} vs overall {std(gc):.2f}')
    else:
        print(f'         question-conditioned: {len(set(qs[i] for i in ok))} distinct questions '
              f'over {len(ok)} samples -> question text carries no usable anchor')

    # ---- 2. error decomposition: which factor carries the loss? ----
    d_len = mean([iou(rebuild(pc[i], gl[i], hi_clip), gts[ok[i]]) for i in range(len(ok))])
    d_ctr = mean([iou(rebuild(gc[i], pl[i], hi_clip), gts[ok[i]]) for i in range(len(ok))])
    both = mean([iou(rebuild(gc[i], gl[i], hi_clip), gts[ok[i]]) for i in range(len(ok))])
    print('[decomp] replace one factor with GT (oracle):')
    print(f'         raw {raw:.4f} | pred center + GT length {d_len:.4f} '
          f'| GT center + pred length {d_ctr:.4f} | both GT {both:.4f}')

    # ---- 3. is the predicted spread already calibrated?  pure rescale sweep ----
    mp = mean(pc)
    sweep = [(k, mean([iou(rebuild(mp + k * (c - mp), pl[i], hi_clip), gts[ok[i]])
                       for i, c in enumerate(pc)]))
             for k in [round(0.4 + 0.1 * i, 1) for i in range(31)]]
    bk, bv = max(sweep, key=lambda t: t[1])
    print(f'[scale] rescale center spread about its own mean (pred lengths kept): '
          f'best k={bk:.1f} -> {bv:.4f} (k=1.0 is raw {raw:.4f}, Δ{bv - raw:+.4f})')
    print('        ' + '  '.join(f'{k:.1f}:{v:.3f}' for k, v in sweep[::3]))

    # ---- 4. decode-time corrections ----
    best = (raw, 0)
    for d in range(-25, 26):
        v = mean([iou(rebuild(pc[i] + d, pl[i], hi_clip), gts[ok[i]]) for i in range(len(ok))])
        if v > best[0]:
            best = (v, d)
    aff = mean([iou(rebuild(cc * pc[i] + ic, pl[i], hi_clip), gts[ok[i]])
                for i in range(len(ok))])
    med_len = quantile(ref_l, 0.5)
    gtlen = mean([iou(rebuild(pc[i], med_len, hi_clip), gts[ok[i]]) for i in range(len(ok))])
    cands = [('best global shift δ=%+d [oracle]' % best[1], best[0]),
             ('OLS affine center, slope %.2f [oracle]' % cc, aff),
             ('eval median length (%.0f f) [oracle]' % med_len, gtlen)]

    # Monotone remap of the predicted center onto a reference distribution.  Ties
    # stay tied, so a collapsed mode cannot be rescued by this transform — that is
    # the point: it bounds what ANY rank-preserving decode correction can buy.
    pred_c_sorted = sorted(pc)

    def qmap(ref, use_gt_len=False):
        m = [quantile(ref, midrank_pct(pred_c_sorted, c)) for c in pc]
        return mean([iou(rebuild(m[i], gl[i] if use_gt_len else pl[i], hi_clip), gts[ok[i]])
                     for i in range(len(ok))])

    print(f'[quantile] onto eval GT centers [oracle]: pred len {qmap(ref_c):.4f} '
          f'| GT len {qmap(ref_c, True):.4f}')
    cands.append(('quantile map -> eval GT centers [oracle]', qmap(ref_c, True)))

    honest = None
    if prior:
        pcen, plen = sorted(prior['centers']), sorted(prior['lengths'])
        match = mean([iou(rebuild(mean(prior['centers']) + (c - mp) * std(pcen) / std(pc),
                                  pl[i], hi_clip), gts[ok[i]])
                      for i, c in enumerate(pc)])
        honest = match
        cands += [('quantile map -> train prior [honest]', qmap(pcen)),
                  ('mean+std match to train prior [honest]', match),
                  ('train median length (%.0f f) [honest]' % quantile(plen, 0.5),
                   mean([iou(rebuild(pc[i], quantile(plen, 0.5), hi_clip), gts[ok[i]])
                         for i in range(len(ok))]))]
        # label-free floor: one constant interval, chosen on the TRAIN GT only
        tr = prior.get('starts') and prior.get('ends')
        if tr:
            step = max(1, len(prior['starts']) // 6000)
            t_iv = [(a, b) for a, b in zip(prior['starts'], prior['ends'])][::step]
            t_hi = max(b for _, b in t_iv)
            t_best = (0.0, None)
            for c in range(0, min(t_hi, hi_clip) + 1):
                for L in range(1, 26):
                    iv = rebuild(c, L, t_hi)
                    v = mean([iou(iv, g) for g in t_iv])
                    if v > t_best[0]:
                        t_best = (v, iv)
            iv = t_best[1]
            cands.append(('train-fit constant %s [honest]' % (iv,),
                          mean([iou(iv, gts[ok[i]]) for i in range(len(ok))])))
            print(f'[floor] best constant on TRAIN GT = {t_best[1]} '
                  f'(train IoU {t_best[0]:.4f}, fit on {len(t_iv)} train TG answers)')
        print(f'[prior] train centers mean={mean(prior["centers"]):.2f} std={std(prior["centers"]):.2f} '
              f'median={quantile(pcen, .5):.1f} | eval GT mean={mean(gc):.2f} std={std(gc):.2f} '
              f'median={quantile(ref_c, .5):.1f} | pred mean={mp:.2f} std={std(pc):.2f}')

    # best constant interval: the floor a prior-only model reaches
    cons = (0.0, None)
    for c in range(0, hi_clip + 1):
        for L in range(1, min(hi_clip + 1, 26)):
            v = mean([iou(rebuild(c, L, hi_clip), gts[ok[i]]) for i in range(len(ok))])
            if v > cons[0]:
                cons = (v, (c, L))
    print(f'  {"correction":40s} mIoU      Δ vs raw')
    for name, v in cands:
        print(f'  {name:40s} {v:.4f}  {v - raw:+.4f}')
    print(f'  {"best constant interval %s [oracle]" % (cons[1],):40s} {cons[0]:.4f}  '
          f'{cons[0] - raw:+.4f}   <- prior-only floor')
    return raw, spearman(pc, gc), cc, cons[0], honest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_a')
    ap.add_argument('run_b')
    ap.add_argument('--labels', nargs=2, default=['A', 'B'])
    ap.add_argument('--prior', help='train-set GT prior from tg_prior_from_train.py '
                                    '(dict with "centers", or a bare array of frame indices)')
    a = ap.parse_args()
    prior = None
    if a.prior:
        with open(a.prior, encoding='utf-8') as fh:
            raw_prior = json.load(fh)
        prior = raw_prior if isinstance(raw_prior, dict) else {'centers': raw_prior}
    rows = []
    for path, label in ((a.run_a, a.labels[0]), (a.run_b, a.labels[1])):
        preds, gts, qs, real = load_any(path)
        rows.append((label, real) + probe(preds, gts, qs, label, prior))
    print('\n==== summary ====')
    for label, real, raw, rho, slope, cons, honest in rows:
        hb = f'{honest:.4f}' if honest is not None else 'n/a  '
        print(f'  {label:14s} {os.path.basename(real):18s} mIoU={raw:.4f} rho={rho:+.3f} '
              f'slope={slope:+.3f} prior-floor={cons:.4f} honest-best={hb}')
    print(f'  ΔmIoU {rows[1][2] - rows[0][2]:+.4f} | Δrho {rows[1][3] - rows[0][3]:+.3f} | '
          f'Δslope {rows[1][4] - rows[0][4]:+.3f}')
    print('  判读：rho 高 + 展开类校正能涨 → 信号在、被先验压住，解码端值得做；'
          'rho 差不多而展开收益封顶 → 瓶颈在目标函数，走 RL')
    return 0


if __name__ == '__main__':
    sys.exit(main())
