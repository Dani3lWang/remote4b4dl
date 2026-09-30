"""CPU-only replay of saved masks. No remote writes or model inference."""
import sys
sys.dont_write_bytecode = True
import collections
import datetime
import hashlib
import json
import pathlib
import runpy
import numpy as np

root = pathlib.Path('/root/autodl-tmp/mmb4dl/mllm')
metric_path = root / 'vtimellm/segmentation/metrics.py'
mod = runpy.run_path(str(metric_path))
manifest = [json.loads(line) for line in (root/'reasonseg_data_trainval/reasonseg_val_thin.jsonl').read_text().splitlines() if line.strip()]
records = {(r['sample_token'], r['query']): r for r in manifest}
out = {'captured_at': datetime.datetime.now().astimezone().isoformat(), 'metric_source_sha256': hashlib.sha256(metric_path.read_bytes()).hexdigest(), 'num_classes': 10, 'runs': {}}
for path in sorted((root/'eval_results').glob('reasonseg*')):
    if not (path/'predictions.json').exists(): continue
    pred = json.loads((path/'predictions.json').read_text())
    acc = mod['SegmentationMetricAccumulator'](10)
    gt_hash = hashlib.sha256()
    mask_hash = hashlib.sha256()
    per_scene = collections.defaultdict(lambda: [0, 0])
    per_target = {}
    errors = []
    for row in pred:
        file = path/row['mask_file']
        try:
            mask_hash.update(file.read_bytes())
            with np.load(file, allow_pickle=False) as z:
                pm, pc = z['predicted_masks'].astype(bool), z['predicted_classes'].astype(np.int64)
                tm, tc = z['target_masks'].astype(bool), z['target_classes'].astype(np.int64)
                gt_hash.update(tm.tobytes()); gt_hash.update(tc.tobytes())
                acc.update(pm, pc, tm, tc, token_status=row.get('status','ok'))
                ious, _, _ = mod['pairwise_mask_iou'](pm,tm)
                hits = {j for i,j in mod['optimal_iou_matching'](ious) if ious[i,j] >= .5}
                scene = records.get((row['sample_token'],row['query']),{}).get('scene_token','unknown')
                per_scene[scene][0] += len(hits); per_scene[scene][1] += len(tm)
                for j in range(len(tm)):
                    per_target[str(row['index'])+':'+str(j)] = int(j in hits)
        except Exception as exc:
            errors.append({'index':row.get('index'), 'error':str(exc)})
    out['runs'][path.name] = {'metrics': acc.compute(), 'samples': acc.sample_count,
        'tp':acc.true_positive,'fp':acc.false_positive,'fn':acc.false_negative,
        'negative_samples':acc.no_object_samples,'gt_sha256':gt_hash.hexdigest(),
        'mask_archive_sha256':mask_hash.hexdigest(),'per_scene_hits_targets':dict(per_scene),
        'per_target_hit':per_target,'errors':errors}
a=out['runs'].get('reasonseg-internal679_encfp32')
b=out['runs'].get('reasonseg-a2ext20_encfp32')
if a and b:
    keys=set(a['per_target_hit']) & set(b['per_target_hit'])
    out['paired_control_vs_lossA2']={
        'same_gt':a['gt_sha256']==b['gt_sha256'], 'targets':len(keys),
        'both_hit':sum(a['per_target_hit'][k] and b['per_target_hit'][k] for k in keys),
        'gain':sum(not a['per_target_hit'][k] and b['per_target_hit'][k] for k in keys),
        'loss':sum(a['per_target_hit'][k] and not b['per_target_hit'][k] for k in keys)}
print(json.dumps(out, ensure_ascii=True))
