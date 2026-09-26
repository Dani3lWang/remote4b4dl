#!/usr/bin/env python3
"""Create separate reachable manifests with annotation-verified absent-class queries."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.build_reasonseg_nuscenes import THING_CLASSES, normalize_category


def frame_labels(record, dataroot, lower, upper, categories):
    points = np.fromfile(dataroot / record['lidar_path'], dtype=np.float32).reshape(-1, 5)
    with np.load(dataroot / record['panoptic_path']) as archive:
        labels = np.asarray(archive['data']).reshape(-1)
    if len(points) != len(labels):
        raise ValueError(f"point/label mismatch: {record['sample_token']}")
    inside = ((points[:, :3] >= lower) & (points[:, :3] < upper)).all(1)
    if not inside.any():
        raise ValueError(f"frame has no in-range points: {record['sample_token']}")
    # No size threshold, range crop, or instance-ID filter may turn a present class into a negative.
    semantic_ids = np.unique(labels // 1000).tolist()
    if any(i not in categories for i in semantic_ids):
        raise ValueError('panoptic semantic ID missing from category mapping')
    present = {normalize_category(categories[i]) for i in semantic_ids}
    ids, counts = np.unique(labels, return_counts=True)
    inner_ids, inner_counts = np.unique(labels[inside], return_counts=True)
    inner_counts = dict(zip(inner_ids.tolist(), inner_counts.tolist()))
    partial = {int(i) for i, n in zip(ids, counts) if 0 < inner_counts.get(int(i), 0) < n}
    return set(ids.tolist()), set(inner_ids.tolist()), present, partial


def prepare_records(records, dataroot, categories, point_range, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError('negative fraction must be between zero and one')
    cache, frames, positives = {}, {}, []
    unreachable, affected, partial_kept = 0, 0, 0
    lower, upper = np.asarray(point_range[:3]), np.asarray(point_range[3:])
    for record in records:
        key = record['sample_token']
        identity = (record['scene_token'], record['lidar_path'], record['panoptic_path'])
        if key in frames and identity != frames[key][0]:
            raise ValueError(f'inconsistent frame identity: {key}')
        if key not in cache:
            cache[key] = frame_labels(record, dataroot, lower, upper, categories)
            frames[key] = (identity, record)
        all_ids, inside_ids, _, partial = cache[key]
        targets = record.get('targets', [])
        if any(t['panoptic_id'] not in all_ids for t in targets):
            raise ValueError(f'annotation target absent: {key}')
        bad = sum(t['panoptic_id'] not in inside_ids for t in targets)
        if bad:
            unreachable += bad
            affected += 1
        elif targets:
            positives.append(record)
            partial_kept += sum(t['panoptic_id'] in partial for t in targets)
    candidates = []
    for key in sorted(frames):
        base = frames[key][1]
        for name in THING_CLASSES:
            if name in cache[key][2]:
                continue
            candidates.append({**base, 'query': f'Segment every {name} in the current LiDAR frame.',
                               'answer': '<NOOBJ>', 'targets': []})
    random.Random(seed).shuffle(candidates)
    requested = round(len(positives) * fraction / (1 - fraction))
    if requested > len(candidates):
        raise ValueError(f'need {requested} verified negatives, only {len(candidates)} available')
    output = positives + candidates[:requested]
    random.Random(seed).shuffle(output)
    return output, {
        'records_in': len(records), 'records_out': len(output),
        'positive_records': len(positives), 'negative_records': requested,
        'requested_negative_fraction': fraction,
        'actual_negative_fraction': requested / max(len(output), 1),
        'replaced_existing_negative_records': sum(not r.get('targets') for r in records),
        'unreachable_targets': unreachable, 'dropped_positive_records': affected,
        'partially_out_of_range_targets_kept': partial_kept,
        'negative_policy': 'class absent from all panoptic semantic labels, including tiny/out-of-range/instance-zero points',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-manifest', type=Path, required=True)
    parser.add_argument('--validation-manifest', type=Path, required=True)
    parser.add_argument('--test-manifest', type=Path, required=True)
    parser.add_argument('--dataroot', type=Path, required=True)
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--negative-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=20260917)
    args = parser.parse_args()
    paths = [args.train_manifest, args.validation_manifest, args.test_manifest]
    if args.output_dir.exists():
        raise ValueError('output directory must be new; never overwrite an active manifest')
    rows = [[json.loads(line) for line in path.read_text().splitlines() if line.strip()] for path in paths]
    scenes = [{r['scene_token'] for r in group} for group in rows]
    if any(scenes[i] & scenes[j] for i in range(3) for j in range(i)):
        raise ValueError('train/dev/test scenes overlap')
    category_rows = json.loads((args.dataroot / args.version / 'category.json').read_text())
    categories = {int(r['index']): r['name'] for r in category_rows}
    point_range = [-51.2, -51.2, -5., 51.2, 51.2, 3.]
    outputs, reports = [], []
    for index, (path, group) in enumerate(zip(paths, rows)):
        output, report = prepare_records(group, args.dataroot, categories, point_range,
                                         args.negative_fraction, args.seed + index)
        encoded = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in output)
        report.update(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      output_sha256=hashlib.sha256(encoded.encode()).hexdigest(), filename=path.name)
        outputs.append((path.name, encoded)); reports.append(report)
    args.output_dir.mkdir(parents=True)
    for name, encoded in outputs:
        (args.output_dir / name).write_text(encoded, encoding='utf-8')
    report = {'seed': args.seed, 'point_cloud_range': point_range, 'scene_overlap': 0,
              'evaluation_protocol': 'derived reachable + negative diagnostic sets; not original official full evaluation',
              'manifests': reports}
    (args.output_dir / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
