"""One immutable, serial evaluation job; never starts or resumes training."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path('/root/autodl-tmp/mmb4dl-core-sft-20261023-r1')
SOURCE = ROOT/'source-1d23109'
OUT = ROOT/'evaluation-20261023-r1'
DATA = '/root/autodl-tmp/mmb4dl-core-trainval-20261023-r1/trainval-v1'
BASE = str(SOURCE/'mllm/configs/core_geometry_baseline_original_b3_trainval.json')
PREP = str(SOURCE/'mllm/configs/core_geometry_preparation.json')
PLAN = str(SOURCE/'mllm/configs/core_geometry_experiment_sft_v1.json')
PROTO = str(ROOT/'protocol-v2/six-task-manifest-final.json')
CKPT = str(ROOT/'training/A3-seed42/step-4750.pt')
os.chdir(SOURCE)
os.environ.update(PYTHONPATH=str(SOURCE/'mllm'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
OUT.mkdir(exist_ok=True)
status = {'status':'running', 'pid':os.getpid(), 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'training_complete':False, 'evaluated_checkpoint_step':4750, 'completed_stages':[],
    'source':str(SOURCE), 'six_task_oracle_assisted_metadata':True}
def save():
    temp = OUT/'status.tmp'
    temp.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    temp.replace(OUT/'status.json')
def run(name, args):
    status['current_stage']=name; status['command']=[sys.executable]+args; save()
    print('START',name,flush=True)
    with (OUT/(name+'.log')).open('x') as stream:
        result = subprocess.run([sys.executable]+args,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(name+' exited '+str(result.returncode))
    status['completed_stages'].append(name); save()
    print('DONE',name,flush=True)
common = ['--dataset',DATA,'--baseline',BASE,'--preparation',PREP,'--experiment',PLAN]
try:
    run('audit-training',[str(OUT/'audit_training.py')])
    # Full dev is first; test endpoint was selected before examining any score.
    for split in ('internal_dev','test'):
        for arm in ('A3','A0'):
            dest = OUT/(arm+'-'+split)
            run(arm+'-'+split,['-m','b4dl_core.b3','infer']+common+['--arm',arm,'--split',split,
                '--batch-size','1','--max-new-tokens','256','--output',str(dest)]+
                (['--checkpoint',CKPT] if arm=='A3' else []))
            run(arm+'-'+split+'-metrics',['-m','b4dl_core','evaluate','--dataset',DATA,'--split',split,
                '--predictions',str(dest/'predictions.jsonl'),'--run',str(dest/'run.json'),
                '--output',str(OUT/(arm+'-'+split+'-report.json'))])
        run(split+'-comparison',['-m','b4dl_core','compare','--base',str(OUT/('A0-'+split+'-report.json')),
            '--candidate',str(OUT/('A3-'+split+'-report.json')),'--output',str(OUT/(split+'-comparison.json'))])
        run(split+'-prior',['-m','b4dl_core','prior','--dataset',DATA,'--split',split,
            '--policy','train_majority','--output',str(OUT/(split+'-prior.json'))])
    for arm in ('A0','A3'):
        dest = OUT/(arm+'-six-task')
        run(arm+'-six-task',['-m','b4dl_core.six_task','infer']+common+['--protocol',PROTO,'--arm',arm,
            '--output',str(dest)]+(['--checkpoint',CKPT] if arm=='A3' else []))
        run(arm+'-six-task-metrics',['-m','b4dl_core.six_task','metrics','--dataset',DATA,'--protocol',PROTO,
            '--predictions',str(dest/'predictions.jsonl'),'--output',str(OUT/(arm+'-six-task-metrics'))])
    status.update(status='evaluation_complete',current_stage=None,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
except Exception as error:
    status.update(status='evaluation_failed',error=str(error),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    traceback.print_exc()
    save()
    sys.exit(1)
save()
