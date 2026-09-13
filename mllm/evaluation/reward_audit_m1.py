#!/usr/bin/env python
"""M1: 零 GPU 的 RL 奖励离线审计（GRPO 立项前的必过门）。

不训练、不占卡，只回答四件事：
  S1 奖励函数直接复用 evaluate_model 的方法时，语义是否如我们所愿（单元用例 + 真实
     预测里的畸形答案占比）；
  S2 奖励在真实预测上的分布与退化率（全 0 / 全 1 / 取值分辨率）；
  S3 先验地板（只用训练集先验算，绝不碰测试标签）与"奖励黑客面"——有多少样本躺着
     输出先验常数就能拿到不低于模型自己的奖励；
  S4 GRPO 组内优势可行性：用观测到的答案分布做代理，估 G 个 rollout 里优势全为 0
     （组内奖励无方差 → 无梯度）的比例，并比较"减地板"前后。
  S5 oracle 红线与训练/测试泄漏的静态检查。

S4 的散布是**代理假设**（按 B3 自身答案的经验分布抽样），不是真 rollout 的采样方差，
只用于判定"值不值得上卡"，不能当作 RL 的预测结果。

用法（服务器，wqlc python，纯 CPU）：
    python evaluation/reward_audit_m1.py \
        --run eval_results/migration_b3 --run eval_results/framepos3ep \
        --train b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json \
        --test  b4dl_dataset/test_qa.json \
        --out   eval_results/reward_baseline.json
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_model import B4DLEvaluator  # noqa: E402  奖励必须复用评测同一份代码

EV = B4DLEvaluator()
ACC_TASKS = ('existence', 'binary_qa')
SEED = 20260913


def reward_tg(pred, gt):
    """逐样本 IoU；解析失败 → 0.0（与 compute_miou 同义）。"""
    return EV.compute_miou([EV.extract_time_grounding_frames(pred or '')],
                           [EV.extract_time_grounding_frames(gt or '')])


def reward_acc(pred, gt):
    return EV.compute_accuracy([pred or ''], [gt or ''])


REWARD = {'time_grounding': reward_tg,
          'existence': reward_acc, 'binary_qa': reward_acc}


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def load_run(d):
    with open(os.path.join(d, 'predictions.json'), encoding='utf-8') as fh:
        preds = json.load(fh)
    met = os.path.join(d, 'metrics.json')
    metrics = json.load(open(met, encoding='utf-8')) if os.path.isfile(met) else None
    return preds, metrics


# ---------------------------------------------------------------- S1 单元语义
UNIT = [
    ('完全一致', 'from frame 006 to frame 014.', 'from frame 006 to frame 014.'),
    ('完全不相交', 'from frame 030 to frame 040.', 'from frame 006 to frame 014.'),
    ('顺序颠倒', 'from frame 014 to frame 006.', 'from frame 006 to frame 014.'),
    ('部分重叠', 'from frame 010 to frame 018.', 'from frame 006 to frame 014.'),
    ('被包含', 'from frame 008 to frame 012.', 'from frame 006 to frame 014.'),
    ('单帧', 'from frame 006 to frame 006.', 'from frame 006 to frame 014.'),
    ('越界超长', 'from frame 000 to frame 099.', 'from frame 006 to frame 014.'),
    ('无法解析', 'I cannot tell from the given frames.', 'from frame 006 to frame 014.'),
    ('空串', '', 'from frame 006 to frame 014.'),
]


def s1_unit(preds_by_task):
    print('\n[S1] 奖励函数单元语义（直接调 evaluate_model，未复制实现）')
    print(f'  {"case":12s} reward   备注')
    for name, p, g in UNIT:
        r = reward_tg(p, g)
        note = ''
        if name == '顺序颠倒' and r == 0.0:
            note = '帧号正确但顺序反 → 0，RL 里这类 rollout 白扔'
        if name == '无法解析':
            note = '符合预期：不可解析必须 0，不能记成 frame 0'
        print(f'  {name:12s} {r:.4f}   {note}')
    print('  accuracy 语义: "Yes." vs "yes" -> %.1f | "The answer is yes" vs "yes" -> %.1f'
          % (reward_acc('Yes.', 'yes'), reward_acc('The answer is yes', 'yes')))
    print('  accuracy 语义: "yes, there is a car." vs "yes" -> %.1f  <- 啰嗦即判错'
          % reward_acc('yes, there is a car.', 'yes'))

    tg = preds_by_task.get('time_grounding')
    if tg:
        bad = [EV.extract_time_grounding_frames(p) for p in tg['predictions']]
        rev = sum(1 for x in bad if x and x['start_frame'] > x['end_frame'])
        unp = sum(1 for x in bad if x is None)
        print(f'  真实预测里的畸形 TG 答案: 顺序颠倒 {rev}/{len(bad)} '
              f'({rev/len(bad):.2%}) | 无法解析 {unp}/{len(bad)} ({unp/len(bad):.2%})')
    for t in ACC_TASKS:
        if t not in preds_by_task:
            continue
        ps = preds_by_task[t]['predictions']
        norm = Counter(EV.normalize_answer(p) for p in ps)
        yesno = norm['yes'] + norm['no']
        verbose = [(k, v) for k, v in norm.most_common(8) if k not in ('yes', 'no')]
        print(f'  {t}: yes/no 占 {yesno}/{len(ps)} ({yesno/len(ps):.1%})，'
              f'其余非 yes/no 答案 {len(ps)-yesno} 条一律记 0；高频异形 {verbose[:4]}')


# ---------------------------------------------------------- S2 分布与退化率
def s2_distribution(runs, floors):
    print('\n[S2] 真实预测上的奖励分布与退化率（reward = 逐样本指标，与评测同源）')
    print(f'  {"run/task":34s} {"n":>5s} {"mean":>7s} {"=0":>6s} {"=1":>6s} '
          f'{"distinct":>8s} {"floor":>6s} {"mean-floor":>10s}')
    for tag, preds in runs:
        for t in ('time_grounding', 'existence', 'binary_qa'):
            if t not in preds:
                continue
            fn = REWARD[t]
            rs = [fn(p, g) for p, g in zip(preds[t]['predictions'], preds[t]['ground_truths'])]
            fl = floors.get(t, 0.0)
            print(f'  {tag + "/" + t:34s} {len(rs):5d} {mean(rs):7.4f} '
                  f'{sum(1 for r in rs if r == 0)/len(rs):6.1%} '
                  f'{sum(1 for r in rs if r == 1)/len(rs):6.1%} '
                  f'{len(set(round(r, 4) for r in rs)):8d} {fl:6.4f} '
                  f'{mean(rs)-fl:+10.4f}')


# ---------------------------------------------------- S3 先验地板与黑客面
def s3_floor(train_path, test_path, runs):
    print('\n[S3] 先验地板（只用训练集，不碰测试标签）')
    with open(train_path, encoding='utf-8') as fh:
        train = json.load(fh)
    tg_iv = []
    yesno = Counter()
    for x in train:
        ans = x['conversations'][1]['value']
        iv = EV.extract_time_grounding_frames(ans)
        if iv:
            tg_iv.append((iv['start_frame'], iv['end_frame']))
        n = EV.normalize_answer(ans)
        if n in ('yes', 'no'):
            yesno[n] += 1
    hi = max(e for _, e in tg_iv)

    # 单一常数区间在训练集上的最优 IoU —— GRPO 必须超过它才算学到东西
    best = (0.0, None)
    for c in range(0, hi + 1):
        for L in range(1, 26):
            s, e = c, c + L - 1
            v = mean([max(0, min(e, b) - max(s, a) + 1) /
                      ((e - s + 1) + (b - a + 1) - max(0, min(e, b) - max(s, a) + 1))
                      if min(e, b) - max(s, a) + 1 > 0 else 0.0 for a, b in tg_iv[::7]])
            if v > best[0]:
                best = (v, (s, e))
    floors = {'time_grounding': best[0],
              'existence': max(yesno.values()) / max(sum(yesno.values()), 1),
              'binary_qa': max(yesno.values()) / max(sum(yesno.values()), 1)}
    print(f'  训练集 TG 答案 {len(tg_iv)} 条；最优常数区间 from frame {best[1][0]:03d} '
          f'to frame {best[1][1]:03d} → 训练集自测 IoU {best[0]:.4f}')
    print(f'  训练集 yes/no 答案 {sum(yesno.values())} 条，多数类占比 '
          f'{floors["existence"]:.4f}（{yesno.most_common(1)}）')
    print('  ⚠ 训练集只有 time_grounding 条目带 task 字段，故 0.6171 是 yes/no 答案空间的'
          '多数类占比，不是 existence 的分任务地板（existence 预测里仅 ~73% 是 yes/no，'
          '其余是 car/bus 等目标名）。要拿分任务地板，RL 数据构建时必须补 task 标签。')

    if test_path and os.path.isfile(test_path):
        with open(test_path, encoding='utf-8') as fh:
            test = json.load(fh)
        ts = {x.get('scene_token') for x in test}
        rs = {x.get('scene_token') for x in train}
        print(f'  泄漏检查: 测试集 {len(ts)} 个 scene_token，训练集 {len(rs)} 个，'
              f'交集 {len(ts & rs)} 个 -> '
              f'{"按 scene 划分成立" if not (ts & rs) else "有重叠，RL 训练集必须剔除"}')

    # 奖励黑客面：躺着输出先验常数能拿到不低于模型自己的奖励的样本占比
    const = 'from frame %03d to frame %03d.' % best[1]
    maj = yesno.most_common(1)[0][0] if yesno else 'yes'
    print('  奖励黑客面（输出先验常数 vs 模型自己的答案，逐样本比较）')
    for tag, preds in runs:
        for t, constans in (('time_grounding', const), ('existence', maj), ('binary_qa', maj)):
            if t not in preds:
                continue
            fn = REWARD[t]
            gts = preds[t]['ground_truths']
            rs = [fn(p, g) for p, g in zip(preds[t]['predictions'], gts)]
            rc = [fn(constans, g) for g in gts]
            better = sum(1 for a, b in zip(rc, rs) if a > b + 1e-9)
            tie = sum(1 for a, b in zip(rc, rs) if abs(a - b) <= 1e-9)
            print(f'    {tag:14s}/{t:16s} 常数赢 {better/len(rs):6.1%}  平手 '
                  f'{tie/len(rs):6.1%}  模型赢 {1-(better+tie)/len(rs):6.1%}')
    return floors, const, maj


# ----------------------------------------------- S4 组内优势可行性（代理）
def s4_advantage(runs, floors, const_tg, const_acc, gs=(4, 8, 16)):
    print('\n[S4] GRPO 组内优势可行性（代理模拟：按该模型自身答案的经验分布抽 G 个 rollout）')
    print('     组内奖励无方差 → advantage 全 0 → 该 prompt 这一步没有梯度')
    rng = random.Random(SEED)
    tag, preds = runs[0]
    for t in ('time_grounding', 'existence', 'binary_qa'):
        if t not in preds:
            continue
        fn = REWARD[t]
        ps, gts = preds[t]['predictions'], preds[t]['ground_truths']
        pool = list(ps)
        distinct = sorted(set(ps))
        fl = floors.get(t, 0.0)
        constans = const_tg if t == 'time_grounding' else const_acc
        print(f'\n  {t} (n={len(ps)}, 代理答案池 {len(pool)} 条 / 去重 {len(distinct)})')
        print(f'    {"G":>3s} {"sampler":22s} {"退化组":>7s} {"E[std r]":>9s} '
              f'{"E[max-min]":>10s} {"E[std r-floor]":>14s}')
        for g in gs:
            for sname, sampler in (('边际经验分布', lambda: rng.choice(pool)),
                                   ('去重后均匀', lambda: rng.choice(distinct)),
                                   ('含常数答案(50%)', lambda: rng.choice(pool)
                                    if rng.random() < 0.5 else constans)):
                deg = 0
                sds, rng_, sds_f = [], [], []
                for i in range(0, len(ps), max(1, len(ps) // 400)):
                    rs = [fn(sampler(), gts[i]) for _ in range(g)]
                    m = mean(rs)
                    sd = (mean([(x - m) ** 2 for x in rs])) ** 0.5
                    rf = [max(x - fl, 0.0) for x in rs]
                    mf = mean(rf)
                    sds.append(sd)
                    sds_f.append((mean([(x - mf) ** 2 for x in rf])) ** 0.5)
                    rng_.append(max(rs) - min(rs))
                    if sd == 0:
                        deg += 1
                print(f'    {g:3d} {sname:22s} {deg/len(sds):7.1%} {mean(sds):9.4f} '
                      f'{mean(rng_):10.4f} {mean(sds_f):14.4f}')


# ------------------------------------------------------------- S5 复用与口径
def s5_reuse(runs):
    print('\n[S0] 复用与同口径复算（奖励必须逐位等于评测指标，否则 RL 在优化别的东西）')
    for tag, preds, metrics in runs:
        if not metrics:
            print(f'  {tag}: 缺 metrics.json，跳过复算')
            continue
        per = metrics.get('per_task_metrics') or metrics.get('per_task') or {}
        fs = metrics.get('final_scores', {})
        rows, worst = [], 0.0
        for t in ('time_grounding', 'existence', 'binary_qa'):
            if t not in preds:
                continue
            rs = [REWARD[t](p, g) for p, g in zip(preds[t]['predictions'],
                                                  preds[t]['ground_truths'])]
            mine = mean(rs)
            ref = None
            if t in per and isinstance(per[t], dict):
                ref = per[t].get('miou' if t == 'time_grounding' else 'accuracy')
            if ref is None and t == 'time_grounding':
                ref = fs.get('miou')
            if ref is not None:
                worst = max(worst, abs(mine - ref))
            rows.append((t, mine, ref))
        txt = '  '.join(f'{t}: audit {m:.6f}' + (f' vs metrics {r:.6f} (Δ{m-r:+.2e})'
                        if r is not None else ' (metrics 无对应项)')
                        for t, m, r in rows)
        gate = 'PASS' if worst <= 1e-6 else f'FAIL(|Δ|={worst:.2e})'
        print(f'  {tag:14s} [{gate}] {txt}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, help='评测结果目录（含 predictions.json）')
    ap.add_argument('--train', default='b4dl_dataset/stage2_full_train_seqv3_meta2_148k.json')
    ap.add_argument('--test', default='b4dl_dataset/test_qa.json')
    ap.add_argument('--out', default='eval_results/reward_baseline.json')
    ap.add_argument('--labels', nargs='*', default=None)
    a = ap.parse_args()

    runs = []
    for i, d in enumerate(a.run):
        preds, metrics = load_run(d)
        label = (a.labels[i] if a.labels and i < len(a.labels)
                 else os.path.basename(d.rstrip('/')))
        runs.append((label, preds, metrics))
    preds_only = [(t, p) for t, p, _ in runs]

    s5_reuse(runs)
    s1_unit(preds_only[0][1] if preds_only else {})
    floors, const_tg, const_acc = s3_floor(a.train, a.test, preds_only)
    s2_distribution(preds_only, floors)
    s4_advantage(preds_only, floors, const_tg, const_acc)

    out = {'note': '仅由训练集先验导出，GRPO 奖励归一化用；不含任何测试标签',
           'source_train': a.train, 'seed': SEED,
           'prior_floor': floors,
           'prior_answer': {'time_grounding': const_tg,
                            'existence': const_acc, 'binary_qa': const_acc},
           'runs': {t: {'mean_reward': {
               k: mean([REWARD[k](p, g) for p, g in
                        zip(v[k]['predictions'], v[k]['ground_truths'])])
               for k in ('time_grounding', 'existence', 'binary_qa') if k in v}}
               for t, v in preds_only}}
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f'\n[OUT] {a.out} 已写出（先验地板 + 先验常数答案 + 各 run 平均奖励）')
    print('  判据：mean_reward(TG) 必须显著高于 prior_floor，否则 RL 的起点就在地板上；')
    print('        退化组比例过高（>30%）说明要么加大 G、要么提高采样温度、要么该任务先不进 RL。')
    print('  奖励形状结论：GRPO 的优势按组内均值中心化，任何**加性**常数（含先验地板）在优势里')
    print('        自动抵消 → "减地板"对梯度是 no-op；而"裁到地板以上"是非线性的，会压掉组内方差')
    print('        （对比 S4 最后两列 E[std r] vs E[std r-floor]）→ 奖励就用原始 IoU / 0-1 准确率。')
    print('        S4 的"含常数答案(50%)"行说明 rollout 越贴近先验众数、组内方差越小，故 RL 初始化')
    print('        取熵更高的那份 checkpoint（见 tg_decode_probe 的中心熵与众数占比）。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
