"""RL data: reuse the SFT training JSON verbatim, never re-render it.

Each entry already carries the injected `<meta>` prompt, the GT answer, the task
label and the per-sequence frame attribution (`feat_indices`), so the rollout
input is byte-identical to what SFT trained on.

Two facts measured on `stage2_full_train_seqv3_meta2_148k.json` drive the layout:
  * only time_grounding entries carry a `task` field (13,124 / 148,271), so
    per-task reward routing needs labels added before any other task joins;
  * `len(feat_indices)` takes just 8 distinct values (3..10, N=5 covers 61%),
    which makes the equal-N batching below cheap instead of a fragmentation
    problem.
"""

import json
import os
import random
from collections import Counter, defaultdict

import torch

from evaluation.test_b4dl import load_features, slice_features

from . import DEFAULT_FEAT_FOLDER, DEFAULT_TRAIN_JSON

TASKS_WITH_REWARD = ('time_grounding', 'existence', 'binary_qa')


class FeatureStore:
    """Scene feature cache. The whole corpus is ~100MB, so keep it resident.

    Reuses evaluation.test_b4dl.load_features/slice_features so the tensor and
    the frame selection are identical to the frozen evaluation path.
    """

    def __init__(self, feat_folder=DEFAULT_FEAT_FOLDER):
        self.feat_folder = feat_folder
        self._scenes = {}

    def scene(self, scene_id):
        feat = self._scenes.get(scene_id)
        if feat is None:
            feat = load_features(self.feat_folder, scene_id)
            self._scenes[scene_id] = feat
        return feat

    def get(self, entry):
        """(N, 768) fp16 CPU tensor for this entry's own frame attribution."""
        return slice_features(self.scene(entry['scene_id']), entry['feat_indices'])

    def __len__(self):
        return len(self._scenes)


def load_entries(path=DEFAULT_TRAIN_JSON, tasks=('time_grounding',)):
    """Flatten the training JSON into rollout-ready entries.

    Entries without a `task` field are dropped and counted: an unlabelled entry
    cannot be routed to a reward function, and guessing the task from the
    question text is exactly the heuristic routing the plan rejected.
    """
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    entries, skipped = [], Counter()
    for x in data:
        task = x.get('task')
        if task is None:
            skipped['no_task_label'] += 1
            continue
        if tasks and task not in tasks:
            skipped[f'task={task}'] += 1
            continue
        feat_indices = x.get('feat_indices')
        if not feat_indices:
            skipped['no_feat_indices'] += 1
            continue
        conv = x['conversations']
        if len(conv) < 2 or conv[0]['from'] != 'human' or conv[1]['from'] != 'gpt':
            skipped['bad_conversations'] += 1
            continue
        entries.append({
            'scene_token': x.get('scene_token'),
            'scene_id': x['scene_id'],
            'task': task,
            'prompt': conv[0]['value'],
            'gt': conv[1]['value'],
            'feat_indices': [int(i) for i in feat_indices],
            'n_visual': len(feat_indices),
        })
    return entries, skipped


def group_stats(entries):
    return {'n': len(entries),
            'by_n_visual': dict(sorted(Counter(e['n_visual'] for e in entries).items())),
            'by_task': dict(Counter(e['task'] for e in entries)),
            'scenes': len({e['scene_token'] for e in entries})}


def buckets_by_n_visual(entries):
    out = defaultdict(list)
    for e in entries:
        out[e['n_visual']].append(e)
    return dict(out)


def iter_prompt_batches(entries, prompts_per_step, repeat=1, shuffle=True, seed=0):
    """Yield lists of entries that all share one visual-token count.

    Equal N is a correctness requirement for generation, not an optimisation:
    arch.py:89-94 rebuilds the attention mask at every decoding step by
    appending ones until it reaches cache_len+1. Those ones land on the real
    positions only when every row of the batch expanded by the same amount, so
    with mixed N the shorter rows get a wrong mask and wrong position_ids --
    they silently generate from a corrupted state instead of erroring.

    `repeat` puts each prompt in the batch that many times, which is how a GRPO
    group of G samples is drawn in one generate call.
    """
    buckets = buckets_by_n_visual(entries)
    rng = random.Random(seed)
    for n_visual in sorted(buckets):
        group = list(buckets[n_visual])
        if shuffle:
            rng.shuffle(group)
        for i in range(0, len(group), prompts_per_step):
            batch = group[i:i + prompts_per_step]
            if len(batch) < prompts_per_step:
                continue
            yield [e for e in batch for _ in range(repeat)]


def write_jsonl(entries, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + '\n')
    return path


def read_jsonl(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]
