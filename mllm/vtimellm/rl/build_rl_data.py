"""M2.1 -- extract the RL subset from the SFT training JSON.

Writes a compact JSONL so a training step never re-parses the 137MB source, and
reports how many entries were dropped for missing a `task` label (the
prerequisite the M1 audit surfaced: only time_grounding is labelled, so other
tasks cannot be routed to a reward until labels are backfilled).

    python -m vtimellm.rl.build_rl_data --tasks time_grounding \
        --out b4dl_dataset/rl_tg_train.jsonl
"""

import argparse
import json

from . import DEFAULT_TRAIN_JSON
from .data import FeatureStore, group_stats, load_entries, write_jsonl


def verify_n_visual(entries, store):
    """Recount n_visual from the real feature slice.

    slice_features drops out-of-range frame indices, so len(feat_indices) can
    overstate the number of visual tokens a batch will actually expand to.
    Equal N per batch is a correctness requirement (arch.py:89-94), so the count
    written to disk has to be the one the tensor really has.
    """
    fixed, missing = 0, 0
    kept = []
    for e in entries:
        try:
            feat = store.get(e)
        except (FileNotFoundError, ValueError):
            missing += 1
            continue
        real = int(feat.shape[0])
        if real != e['n_visual']:
            fixed += 1
            e['n_visual'] = real
        kept.append(e)
    return kept, fixed, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEFAULT_TRAIN_JSON)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tasks', nargs='*', default=['time_grounding'])
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--feat-folder', default=None)
    a = ap.parse_args()

    entries, skipped = load_entries(a.src, tasks=tuple(a.tasks) or None)
    if a.limit:
        entries = entries[:a.limit]

    store = FeatureStore(a.feat_folder) if a.feat_folder else FeatureStore()
    entries, fixed, missing = verify_n_visual(entries, store)
    if missing:
        skipped['features_unreadable'] = missing

    write_jsonl(entries, a.out)

    print(f'[M2.1] {a.src} -> {a.out}')
    print(f'  kept {len(entries)} entries; dropped {dict(skipped) or "none"}')
    print(f'  n_visual recounted from features: {fixed} corrected, '
          f'{missing} dropped, {len(store)} scenes cached')
    print(f'  stats {json.dumps(group_stats(entries), ensure_ascii=False)}')
    for e in entries[:2]:
        print(f'  sample scene={e["scene_id"]} N={e["n_visual"]} '
              f'feat_indices={e["feat_indices"]}')
        print(f'    prompt[:110] {e["prompt"][:110]!r}')
        print(f'    gt           {e["gt"]!r}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
