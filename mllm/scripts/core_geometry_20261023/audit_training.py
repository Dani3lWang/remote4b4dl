import json
from pathlib import Path
import hashlib
import math
import statistics
import torch
from b4dl_core.inputs import GeometryDataset
from b4dl_core.b3 import batch_indices, identity_for
from types import SimpleNamespace

root = Path('/root/autodl-tmp/mmb4dl-core-sft-20261023-r1')
out = root/'evaluation-20261023-r1'
out.mkdir(exist_ok=True)
training = root/'training/A3-seed42'
latest = json.loads((training/'latest.json').read_text())
path = training/latest['checkpoint']
payload = torch.load(path, map_location='cpu', weights_only=True)
def tensors(value, prefix=''):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for k, v in value.items(): yield from tensors(v, prefix+'/'+str(k))
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value): yield from tensors(v, prefix+'/'+str(i))
checked = list(tensors({'parameters': payload['parameters'], 'optimizer': payload['optimizer']}))
rows = [json.loads(line) for line in (training/'loss.jsonl').read_text().splitlines()]
assert [r['step'] for r in rows] == list(range(1, len(rows)+1))
bins = []
for start in range(0, len(rows), 500):
    part = rows[start:start+500]
    bins.append({'first_step':part[0]['step'], 'last_step':part[-1]['step'],
        'mean_loss':statistics.mean(r['loss'] for r in part),
        'mean_gradient_norm':statistics.mean(r['gradient_norm'] for r in part)})
data = GeometryDataset('/root/autodl-tmp/mmb4dl-core-trainval-20261023-r1/trainval-v1', 'train')
source = root/'source-1d23109'
args = SimpleNamespace(baseline=str(source/'mllm/configs/core_geometry_baseline_original_b3_trainval.json'),
    experiment=str(source/'mllm/configs/core_geometry_experiment_sft_v1.json'))
failed = []
for micro in range(4):
    idx = batch_indices(len(data.requests), len(rows), micro, 1, 4, 42)[0]
    r = data.requests[idx]; item = data.model_input(r)
    failed.append({'micro':micro, 'index':idx, 'request_id':r['request_id'], 'scene_id':r['scene_id'],
        'task':r['task'], 'predicate':r['predicate'], 'feature_abs_max':float(abs(item['features']).max()),
        'feature_shape':list(item['features'].shape), 'target':data.targets[r['request_id']]})
report = {'status':'training_failed', 'error':json.loads((training/'error.json').read_text()),
    'job_exit':json.loads((root/'jobs/A3-seed42/exit.json').read_text()),
    'completion_exists':(training/'completion.json').exists(), 'completed_updates':len(rows),
    'planned_updates':13851, 'progress':len(rows)/13851, 'first':rows[0], 'last':rows[-1],
    'loss_windows':bins, 'latest':latest, 'checkpoint_training':payload['training'],
    'actual_checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'identity_matches':payload['identity']==latest['identity']==identity_for(args,data),
    'parameter_tensor_count':len(payload['parameters']), 'checked_tensor_count':len(checked),
    'nonfinite_tensors':[n for n,t in checked if not torch.isfinite(t).all()],
    'scaler':payload['scaler'], 'adam_steps':sorted({float(s['step']) for s in payload['optimizer']['state'].values()}),
    'failed_update_candidates':failed,
    'limitation':'Failing microbatch is not logged; checkpoint 4750 is not the unsaved state after 4953.'}
assert report['actual_checkpoint_sha256']==latest['checkpoint_sha256']
assert report['identity_matches'] and not report['nonfinite_tensors']
(out/'training-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(out/'loss-snapshot.jsonl').write_bytes((training/'loss.jsonl').read_bytes())
print(json.dumps({k:report[k] for k in ('completed_updates','progress','identity_matches','nonfinite_tensors','adam_steps','loss_windows','failed_update_candidates')},ensure_ascii=False))
