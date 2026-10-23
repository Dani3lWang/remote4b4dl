"""Detached, bounded single-run training launcher with durable process/exit receipts."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from .io import read_json, sha256, write_json


def training_command(args):
    plan = read_json(args.experiment)
    recipe = plan['recipe']
    command = [sys.executable, '-u', '-m', 'b4dl_core.b3', 'train', '--dataset', args.dataset,
        '--baseline', args.baseline, '--preparation', args.preparation, '--experiment', args.experiment,
        '--arm', args.arm, '--seed', str(args.seed), '--output', args.output]
    for key in ('batch_size', 'grad_accum', 'steps', 'lr', 'lora_rank', 'save_every', 'keep_last'):
        command.extend(['--'+key.replace('_', '-'), str(recipe[key])])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('start', 'watch'))
    for name in ('job-dir', 'dataset', 'baseline', 'preparation', 'experiment', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--arm', default='A3', choices=('A1', 'A2', 'A3'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    job = Path(args.job_dir).resolve()
    if args.command == 'start':
        if Path(args.output).exists(): parser.error('training output already exists')
        job.mkdir(parents=True)
        command = [sys.executable, '-u', '-m', 'b4dl_core.job', 'watch']
        for name in ('job_dir', 'dataset', 'baseline', 'preparation', 'experiment', 'output', 'arm', 'seed'):
            command.extend(['--'+name.replace('_', '-'), str(getattr(args, name))])
        with (job/'watcher.log').open('x') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        write_json(job/'launch.json', {'watcher_pid': process.pid, 'watcher_command': command,
            'training_command': training_command(args), 'experiment_sha256': sha256(args.experiment),
            'source_commit': os.environ.get('B4DL_SOURCE_COMMIT'), 'time_utc': datetime.now(timezone.utc).isoformat()})
        print('watcher_pid', process.pid, flush=True)
        return
    command = training_command(args)
    with (job/'training.log').open('x') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        write_json(job/'process.json', {'status': 'training_process_started', 'pid': process.pid,
            'watcher_pid': os.getpid(), 'command': command, 'time_utc': datetime.now(timezone.utc).isoformat()})
        return_code = process.wait()
    completed = Path(args.output)/'completion.json'
    write_json(job/'exit.json', {'return_code': return_code,
        'status': 'training_complete' if return_code == 0 and completed.exists() else 'training_failed',
        'completion_sha256': sha256(completed) if completed.exists() else None,
        'time_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__': main()
