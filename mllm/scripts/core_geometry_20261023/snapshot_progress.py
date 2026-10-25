"""Read-only provisional scoring; never presents a prefix as full evaluation."""
import collections
import datetime
import json
from pathlib import Path
import time
from b4dl_core.inputs import GeometryDataset
from b4dl_core.rewards import score_response
from b4dl_core.evaluation import aggregate
from b4dl_core.io import sha256

root = Path('/root/autodl-tmp/mmb4dl-core-sft-20261023-r1/evaluation-20261023-r1')
dataset = GeometryDataset('/root/autodl-tmp/mmb4dl-core-trainval-20261023-r1/trainval-v1', 'internal_dev')
path = root/'A3-internal_dev/predictions.jsonl'
for attempt in range(30):
    try:
        content = path.read_text()
        predictions = [json.loads(line) for line in content.splitlines()]
        if not predictions: raise ValueError('writer currently replacing snapshot')
        break
    except (ValueError, FileNotFoundError): time.sleep(.1)
else: raise RuntimeError('could not obtain a complete prediction snapshot')
assert len(predictions) <= len(dataset.requests)
rows = []
for pred, request in zip(predictions, dataset.requests):
    assert pred['request_id']==request['request_id'] and pred['input_fingerprint']==request['input_fingerprint']
    rows.append({'request_id':request['request_id'],'task':request['task'],'scene_id':request['scene_id'],
        'truth':dataset.targets[request['request_id']]['answer'], 'status':pred['status'],
        **score_response(request,dataset.targets[request['request_id']],pred['text'],pred['status'])})
result = {'status':'partial_progress_only','captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'n_observed':len(rows),'n_full_split':len(dataset.requests),'checkpoint_step':4750,
    'observed_prefix_summary':aggregate(rows),'by_task':{t:aggregate([r for r in rows if r['task']==t]) for t in sorted({r['task'] for r in rows})},
    'limitations':['ID-ordered prefix only; not a representative or complete evaluation.',
        'Original B3 control has not run; no claim of improvement.',
        'No full-split accuracy, bootstrap, opposite-pair or option-orbit claim.']}
(root/'progress-snapshot.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(root/'progress-predictions.jsonl').write_text(content)
print(json.dumps(result,ensure_ascii=False))
