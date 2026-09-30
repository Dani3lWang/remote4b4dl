"""Read-only remote evidence collector; run via SSH stdin, no GPU/model loading."""
import collections
import datetime
import hashlib
import json
import pathlib
import re
import socket
import subprocess

ROOT = pathlib.Path('/root/autodl-tmp/mmb4dl')
M = ROOT / 'mllm'
out = {'host': socket.gethostname(), 'captured_at': datetime.datetime.now().astimezone().isoformat(),
       'root': str(ROOT), 'files': {}, 'manifests': {}, 'errors': []}
out['git_head'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
out['git_history_seg'] = subprocess.check_output(['git', '-C', str(ROOT), 'log', '--all', '--date=iso-strict', '--format=%h %ad %s', '--', 'mllm/vtimellm/segmentation', 'mllm/scripts/train_reasonseg.py'], text=True)

def compact(x):
    if isinstance(x, dict):
        return {k: compact(v) for k, v in x.items() if k not in ('object_metrics', 'per_object', 'per_sample')}
    if isinstance(x, list):
        return [compact(v) for v in x]
    return x

def collect(p):
    p = pathlib.Path(p)
    key = str(p)
    if not p.is_file() or key in out['files']:
        return
    try:
        raw = p.read_bytes()
        ent = {'bytes': len(raw), 'mtime': datetime.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(),
               'sha256': hashlib.sha256(raw).hexdigest()}
        if p.suffix == '.json':
            data = json.loads(raw)
            if p.name == 'predictions.json':
                rows = data if isinstance(data, list) else data.get('predictions', [])
                ids = [(r.get('sample_token'), r.get('query')) for r in rows]
                ent['prediction_summary'] = {'count': len(rows), 'identity_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                                             'first_row_keys': list(rows[0]) if rows else []}
            else:
                ent['json'] = compact(data)
        elif p.suffix in ('.log', '.out', '.txt'):
            text = raw.decode('utf-8', errors='replace')
            text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
            lines = text.replace('\r', '\n').splitlines()
            records = []
            for candidate in re.findall(r'\{[^{}]*\}', text):
                try:
                    rec = json.loads(candidate)
                    if isinstance(rec, dict) and any(k in rec for k in ('teacher_forcing_mean_iou', 'miou', 'loss')):
                        records.append(rec)
                except Exception:
                    pass
            ent['metric_records'] = [r for r in records if 'teacher_forcing_mean_iou' in r or 'miou' in r]
            losses = [r for r in records if 'loss' in r]
            ent['loss_summary'] = {'records': len(losses), 'first': losses[:2], 'last': losses[-2:]}
            selected = []
            for line in lines:
                if re.search(r'teacher_forcing|"validation"|\beval_loss\b|^epoch |^\s*epoch\s*[=:]|^Epoch |"miou"|"point_accuracy"|NaN|nan|No space|RuntimeError|Traceback|Error:|rc=|EXIT=|CHAIN DONE|selected epoch|trainable:|records:|train records|train samples|steps_per|early stop|Early stop|^\{.*"epoch"', line):
                    if not re.search(r'\d+%\||"step"\s*:', line) and len(line) < 2000:
                        selected.append(line)
            ent['selected_lines'] = selected[-300:]
            ent['tail'] = '\n'.join(lines[-18:])[-14000:]
        else:
            ent['text'] = raw.decode('utf-8', errors='replace')
        out['files'][key] = ent
    except Exception as e:
        out['errors'].append({'path': key, 'error': str(e)})

for p in (M / 'eval_results').glob('**/*.json'):
    rel = str(p.relative_to(M / 'eval_results'))
    if any(t in rel for t in ('reasonseg', '_oracle', '_multitask', '_temporal', '_grounding', '_multiframe', '_dtypeab')) or p.name.startswith('metrics'):
        collect(p)
for p in (M / 'training_logs').glob('**/*'):
    if p.suffix in ('.json', '.log', '.txt', '.sh') and not p.name.startswith('stage1_') and not p.name.startswith('stage2_'):
        if any(t in str(p) for t in ('phase2', 'temporal', 'verify_v123', 'tfval', 'tvenc', 'grounding', 'chain_overnight', '_stage', '_smoke')):
            collect(p)
for p in (M / 'checkpoints').glob('**/*.json'):
    if ('reasonseg' in str(p) or '_protected' in str(p)) and p.name in ('trainer_state.json', 'reasonseg_config.json', 'spatial_config.json'):
        collect(p)
for pattern in ('seg*.log', 'spat*.log', 'spatial*.log', 'probe*.log', 'reasonseg*.log', 'diag_eval.out', 'run_seg*.sh', 'run_spatial*.sh'):
    for p in pathlib.Path('/root').glob(pattern):
        collect(p)
for p in pathlib.Path('/root/backup_20260920').glob('**/*'):
    if p.suffix in ('.json', '.log', '.sh') and not p.name.startswith('scenes_'):
        collect(p)
for d in M.glob('reasonseg*'):
    if d.is_dir():
        for p in d.glob('**/*'):
            if p.suffix in ('.log', '.sh') or (p.suffix == '.json' and not p.name.startswith('scenes_')):
                collect(p)
        for p in d.glob('*.jsonl'):
            try:
                n = neg = targets = 0
                scenes = set()
                first = []
                sha = hashlib.sha256()
                for line in p.open('rb'):
                    sha.update(line)
                    if not line.strip(): continue
                    r = json.loads(line)
                    n += 1
                    scenes.add(r.get('scene_token'))
                    neg += not bool(r.get('targets'))
                    targets += len(r.get('targets', []))
                    if n <= 400: first.append(r.get('sample_token'))
                out['manifests'][str(p)] = {'records': n, 'scenes': len(scenes), 'negatives': neg, 'targets': targets, 'first400_unique_frames': len(set(first)), 'sha256': sha.hexdigest()}
            except Exception as e:
                out['errors'].append({'path': str(p), 'error': str(e)})
def json_safe(value):
    # Preserve failed-run NaN/Infinity evidence as strings in strict JSON.
    if isinstance(value, float):
        import math
        if not math.isfinite(value):
            return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value

print(json.dumps(json_safe(out), ensure_ascii=True, allow_nan=False))
