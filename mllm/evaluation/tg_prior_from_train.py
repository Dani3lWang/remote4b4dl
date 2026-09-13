#!/usr/bin/env python
"""Dump the time_grounding GT prior of a *training* file, for the honest (label-free
at test time) decode corrections in tg_decode_probe.py.

Same interval regex as evaluate_model.compute_miou.  Fails fast on any TG entry whose
answer has no parseable interval, because a silent skip would bias the prior.

Usage (on the server, where the 137MB train json lives):
    python evaluation/tg_prior_from_train.py \
        --data b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json \
        --out  eval_results/tg_prior_train.json
"""
import argparse
import json
import re
import sys
from collections import Counter

TG_PATTERN = re.compile(
    r'from\s+frames?\s+(\d+)\s+to\s+frames?\s+(\d+)'
    r'|frames?\s+(\d+)\s+(?:to|-|and)\s+frames?\s+(\d+)'
    r'|from\s+frames?\s+(\d+)\s+until\s+frames?\s+(\d+)', re.I)


def parse(text):
    m = TG_PATTERN.search(text or '')
    if not m:
        return None
    nums = [int(x) for x in m.groups() if x is not None]
    return (nums[0], nums[1])


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def median(xs):
    v = sorted(xs)
    n = len(v)
    return float('nan') if not v else (v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0)


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json')
    ap.add_argument('--out', default='eval_results/tg_prior_train.json')
    args = ap.parse_args()

    with open(args.data, encoding='utf-8') as fh:
        data = json.load(fh)
    tg = [x for x in data if x.get('task') == 'time_grounding']
    iv = [parse(x['conversations'][1]['value']) for x in tg]
    bad = sum(1 for v in iv if v is None)
    if bad:
        sys.exit(f'fail-fast: {bad}/{len(tg)} 条训练 TG 答案解析不出帧号，先验会有偏')

    starts = [a for a, _ in iv]
    ends = [b for _, b in iv]
    centers = [(a + b) / 2.0 for a, b in iv]
    lengths = [b - a + 1 for a, b in iv]
    out = {'source': args.data, 'n': len(iv), 'centers': centers, 'lengths': lengths,
           'starts': starts, 'ends': ends}
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh)

    print(f'{args.data}: {len(data)} 条总计, time_grounding {len(tg)} 条 -> {args.out}')
    print(f'center mean={mean(centers):.2f} median={median(centers):.1f} std={std(centers):.2f}')
    print(f'length mean={mean(lengths):.2f} median={median(lengths):.0f} std={std(lengths):.2f}')
    b = Counter(s // 5 * 5 for s in starts)
    print('start buckets(x5): ' + ' '.join(f'{k}:{b[k]*100//len(starts)}%'
                                          for k in sorted(b)))
    print('top-5 intervals: ' + ', '.join(f'[{a:02d}-{yy:02d}]x{c}'
                                          for (a, yy), c in Counter(iv).most_common(5)))


if __name__ == '__main__':
    main()
