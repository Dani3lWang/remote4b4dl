#!/usr/bin/env python
"""B4a：TG 高帧段过采样——把 GT 答案起始帧 >= threshold 的 time_grounding 样本复制追加。

背景（B4DL_B3整场景meta修复mIoU超论文_20260906.md §3.3）：B3 的 TG 预测整体偏早
（mean −3.9 帧）且高帧段严重欠覆盖——GT start>=25 的样本在训练/测试集均占 ~15%，
但 B3 预测仅 4% 落在高帧段、30+ 帧为 0。将高帧段样本频率 ×2 是最直接的数据侧干预。

条目级复制是安全的：dataset.py 按 index deepcopy 单条、trainer 侧 shuffle，
条目之间无顺序依赖；复制的是已注入 meta2 的完整条目（含 feat_range/feat_indices）。

用法：
  python scripts/oversample_tg_highframe.py \
      --data b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json \
      --out  b4dl_dataset/stage2_full_train_seqv3_meta2_oversampled_150k.json
"""
import argparse
import copy
import json
import re
import sys
from pathlib import Path

TG_PATTERN = re.compile(r'from\s+frames?\s+(\d+)\s+to\s+frames?\s+(\d+)', re.I)


def gt_start(entry):
    m = TG_PATTERN.search(entry['conversations'][1]['value'])
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json')
    ap.add_argument('--out', default='b4dl_dataset/stage2_full_train_seqv3_meta2_oversampled_150k.json')
    ap.add_argument('--threshold', type=int, default=25, help='GT 起始帧 >= threshold 视为高帧段')
    ap.add_argument('--copies', type=int, default=1, help='每条高帧样本追加的复制份数（1 = 频率×2）')
    ap.add_argument('--expected', type=int, default=1951, help='高帧段预期条数（与 0829 探索统计核对，不符则 fail-fast）')
    args = ap.parse_args()

    src = Path(args.data)
    data = json.load(open(src))
    n_total = len(data)

    tg = [x for x in data if x.get('task') == 'time_grounding']
    unparsed = [x for x in tg if gt_start(x) is None]
    if unparsed:
        sys.exit(f'fail-fast: {len(unparsed)} 条 TG 答案无法解析帧号，先核查数据')

    high = [x for x in tg if gt_start(x) >= args.threshold]
    if len(high) != args.expected:
        sys.exit(f'fail-fast: start>={args.threshold} 样本 {len(high)} 条 != 预期 {args.expected}，数据已变化需人工核对')

    extra = [copy.deepcopy(x) for _ in range(args.copies) for x in high]
    out_data = data + extra
    out = Path(args.out)
    with open(out, 'w') as f:
        json.dump(out_data, f, ensure_ascii=False)

    # 写出后校验：总量、高帧段出现次数、抽查复制条目与源逐字节一致
    check = json.load(open(out))
    hi_in = sum(1 for x in check if x.get('task') == 'time_grounding' and (s := gt_start(x)) is not None and s >= args.threshold)
    assert len(check) == n_total + len(extra), '总量不符'
    assert hi_in == len(high) * (1 + args.copies), f'高帧段出现次数 {hi_in} != {len(high)}x{1 + args.copies}'
    assert check[n_total] == high[0] and check[-1] == high[-1], '追加条目内容与源不一致'

    print(f'source: {src.name} = {n_total} 条')
    print(f'time_grounding: {len(tg)} 条, start>={args.threshold}: {len(high)} 条 (每条 +{args.copies} 复制)')
    print(f'output: {out.name} = {len(check)} 条 (高帧段出现 {hi_in} 次, 占 TG {hi_in}/{len(tg) + len(extra)} 的 TG 子集)')
    buckets = {}
    for x in check:
        if x.get('task') == 'time_grounding':
            s = gt_start(x)
            buckets[s // 5 * 5] = buckets.get(s // 5 * 5, 0) + 1
    print('oversampled start buckets:', dict(sorted(buckets.items())))


if __name__ == '__main__':
    main()
