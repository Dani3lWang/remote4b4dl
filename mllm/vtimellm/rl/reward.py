"""Reward = the frozen evaluation metric, and GRPO group normalisation.

The M1 audit (evaluation/reward_audit_m1.py, section S0) showed these functions
reproduce metrics.json to 2e-15, so training against them optimises the reported
number rather than a proxy. Nothing here re-implements a metric.
"""

from collections import Counter

from evaluation.evaluate_model import B4DLEvaluator

EV = B4DLEvaluator()
ACC_TASKS = ('existence', 'binary_qa')
EPS = 1e-6


def score(task, pred, gt):
    """Per-sample reward: closed-interval IoU for TG, 0/1 exact match otherwise.

    Deliberately raw. Group-relative advantages subtract the group mean, so any
    additive constant (including the ~0.300 prior floor) cancels out and is a
    no-op, while clipping at the floor is non-linear and was measured to cut
    group std by 35-60% (audit section S4).
    """
    if task == 'time_grounding':
        return EV.compute_miou(
            [EV.extract_time_grounding_frames(pred or '')],
            [EV.extract_time_grounding_frames(gt or '')])
    if task in ACC_TASKS:
        return EV.compute_accuracy([pred or ''], [gt or ''])
    raise KeyError(f'no reward defined for task {task!r}')


def score_batch(tasks, preds, gts):
    return [score(t, p, g) for t, p, g in zip(tasks, preds, gts)]


def group_advantage(rewards, group_size, divide_std=True):
    """(P, G) advantages plus a keep mask; zero-variance groups are dropped.

    A dropped group contributes no gradient at all, which is why the degenerate
    rate is a first-class training metric rather than a curiosity.
    """
    if len(rewards) % group_size:
        raise ValueError(f'{len(rewards)} rewards is not a multiple of G={group_size}')
    P = len(rewards) // group_size
    adv = [[0.0] * group_size for _ in range(P)]
    keep = [False] * P
    degenerate = 0
    for p in range(P):
        g = list(rewards[p * group_size:(p + 1) * group_size])
        mean = sum(g) / len(g)
        var = sum((x - mean) ** 2 for x in g) / len(g)
        std = var ** 0.5
        if std <= EPS:
            degenerate += 1
            continue
        keep[p] = True
        denom = (std + EPS) if divide_std else 1.0
        adv[p] = [(x - mean) / denom for x in g]
    return adv, keep, degenerate, P


def parse_failures(tasks, preds):
    """TG answers with no parseable frame interval score 0 and waste a rollout."""
    c = Counter()
    for t, p in zip(tasks, preds):
        if t != 'time_grounding':
            continue
        c['n'] += 1
        if EV.extract_time_grounding_frames(p or '') is None:
            c['unparsed'] += 1
    return c
