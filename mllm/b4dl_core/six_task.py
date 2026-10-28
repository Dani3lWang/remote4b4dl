"""Regression for original six-task prompts with native frame geometry and extensions."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .b3 import chosen_config, device_lease, identity_for, load_original
from .integration import attach, language_model, load_checkpoint
from .inputs import GeometryDataset
from .io import digest, read_json, read_jsonl, sha256, write_json, write_jsonl
from .resume import (append_jsonl, bind_run, output_lease, recover_jsonl,
                     validate_predictions, verify_completion)
from .runtime import runtime_identity


def validate_protocol(path, dataset):
    protocol = read_json(path)
    if protocol['schema'] != 'b4dl-core-six-task-v1' or protocol['protocol_id'] != digest({k: v for k, v in protocol.items() if k != 'protocol_id'}):
        raise ValueError('invalid six-task protocol identity')
    if protocol['dataset_id'] != dataset.manifest['dataset_id']:
        raise ValueError('regression dataset mismatch')
    for name, expected in protocol['files_sha256'].items():
        if sha256(name) != expected: raise ValueError('frozen regression input changed: '+name)
    rows = read_jsonl(protocol['request_path'])
    if len(rows) != protocol['requests'] or dict(Counter(r['task'] for r in rows)) != protocol['task_counts']:
        raise ValueError('six-task denominator changed')
    if len({r['request_id'] for r in rows}) != len(rows): raise ValueError('duplicate regression IDs')
    for row in rows:
        record = dataset.manifest['scenes'][row['scene_id']]
        expected = digest({'query': row['query'], 'scene_bundle_sha256': record['sha256'], 'whole_scene': True})
        if row['input_fingerprint'] != expected or record['split'] != 'test':
            raise ValueError('regression prompt/scene identity mismatch')
    return protocol, rows


def query_batch(model, tokenizer, dataset, row):
    from vtimellm.conversation import conv_templates
    from vtimellm.constants import IMAGE_TOKEN_INDEX
    from vtimellm.mm_utils import tokenizer_image_token
    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], row['query']); conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')[None]
    device = language_model(model).get_input_embeddings().weight.device
    bundle = dataset.scene(row['scene_id'])
    batch = {'input_ids': ids.to(device), 'attention_mask': torch.ones_like(ids, device=device),
        'images': torch.as_tensor(bundle['features'], device=device)[None],
        'frame_times': torch.as_tensor(bundle['frame_times'], device=device)[None],
        'frame_geometry': torch.as_tensor(bundle['frame_geometry'], device=device)[None],
        'frame_valid_mask': torch.ones(1, len(bundle['features']), dtype=torch.bool, device=device)}
    return batch


def prediction_arrays(rows, predictions):
    validate_predictions(rows, predictions, complete=True)
    result = defaultdict(lambda: {'predictions': [], 'ground_truths': [], 'questions': [], 'samples': []})
    for row, prediction in zip(rows, predictions):
        if prediction['request_id'] != row['request_id'] or prediction['input_fingerprint'] != row['input_fingerprint']:
            raise ValueError('prediction order/identity mismatch')
        bucket = result[row['task']]
        bucket['predictions'].append(prediction['text'])
        bucket['ground_truths'].append(row['target'])
        bucket['questions'].append(row['query'].replace('<video>', '').replace('<4DLiDAR>', '').split('<meta>')[0].strip())
        bucket['samples'].append({'request_id': row['request_id'], 'scene_id': row['scene_id'], 'status': prediction['status']})
    return dict(result)


def infer_predictions(output, run, rows, dataset, model, tokenizer, resume=False):
    """Called under the output lease; interrupted appends keep an audited prefix."""
    output = Path(output)
    path = output/'predictions.jsonl'
    if run['generation'].get('do_sample', False):
        raise ValueError('resumable six-task inference requires deterministic decoding')
    bind_run(output/'run.json', run, resume, [path, output/'completion.json'])
    completion = {'status': 'six_task_inference_complete', 'requests': len(rows), 'run': run}
    if verify_completion(output, completion, {'predictions_sha256': path}):
        validate_predictions(rows, read_jsonl(path), complete=True)
        return
    predictions = recover_jsonl(path, lambda values: validate_predictions(rows, values))
    for index in range(len(predictions), len(rows)):
        row = rows[index]
        try:
            tokens = model.generate(**query_batch(model, tokenizer, dataset, row),
                **run['generation'], pad_token_id=tokenizer.eos_token_id)[0].tolist()
            ended = tokenizer.eos_token_id in tokens
            if ended: tokens = tokens[:tokens.index(tokenizer.eos_token_id)]
            prediction = {'status': 'ok' if ended else 'truncated', 'text': tokenizer.decode(tokens, skip_special_tokens=True).strip()}
        except Exception as error:
            if isinstance(error, torch.cuda.OutOfMemoryError): raise
            prediction = {'status': 'error', 'text': '', 'error': type(error).__name__+': '+str(error)}
        prediction.update(request_id=row['request_id'], input_fingerprint=row['input_fingerprint'])
        append_jsonl(path, [prediction])
        if (index+1) % 50 == 0: print('six-task predictions', index+1, '/', len(rows), flush=True)
    validate_predictions(rows, read_jsonl(path), complete=True)
    completion['predictions_sha256'] = sha256(path)
    write_json(output/'completion.json', completion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'infer', 'metrics'))
    for name in ('dataset', 'protocol', 'output'): parser.add_argument('--'+name, required=True)
    for name in ('baseline', 'preparation', 'experiment', 'checkpoint', 'predictions'): parser.add_argument('--'+name)
    parser.add_argument('--arm', choices=('A0', 'A1', 'A2', 'A3'), default='A0')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume and args.command != 'infer': parser.error('resume belongs to infer')
    output = Path(args.output)
    if output.exists() and not args.resume: parser.error('choose a new output directory or explicit resume')
    if args.resume and not (output/'run.json').is_file(): parser.error('resume requires an existing inference identity')
    output.mkdir(parents=True, exist_ok=True)
    dataset = GeometryDataset(args.dataset)
    protocol, rows = validate_protocol(args.protocol, dataset)
    if args.command == 'metrics':
        import os
        os.environ['NLTK_DATA'] = protocol['metrics']['nltk_data']
        if protocol['metrics'].get('java_home'):
            os.environ['JAVA_HOME'] = protocol['metrics']['java_home']
            os.environ['PATH'] = str(Path(protocol['metrics']['java_home'])/'bin')+os.pathsep+os.environ.get('PATH', '')
        import nltk
        nltk.data.path.insert(0, protocol['metrics']['nltk_data'])
        from evaluation.evaluate_model import B4DLEvaluator
        import sys
        if sha256(sys.modules[B4DLEvaluator.__module__].__file__) != protocol['backend_sources']['evaluator']['sha256']:
            raise ValueError('metric implementation differs from frozen protocol')
        predictions = read_jsonl(args.predictions)
        evaluator = B4DLEvaluator(use_gpt=False, bert_model_path=protocol['metrics']['bert_model_path'], meteor_backend='dual')
        results = prediction_arrays(rows, predictions)
        metrics = evaluator.evaluate_all(results)
        evaluator.save_results(metrics, evaluator.compute_final_score(metrics), str(output/'metrics.json'))
        write_json(output/'completion.json', {'status': 'six_task_metrics_complete', 'protocol_id': protocol['protocol_id'],
            'requests': len(rows), 'prediction_statuses': dict(Counter(r['status'] for r in predictions)),
            'predictions_sha256': sha256(args.predictions), 'oracle_assisted_metadata': True})
        return
    if not args.baseline or not args.preparation: parser.error('model commands require baseline and preparation')
    if args.command == 'infer' and ((args.arm == 'A0' and args.checkpoint) or (args.arm != 'A0' and not args.checkpoint)):
        parser.error('A0 uses original B3; extensions require a checkpoint')
    preparation, lock = read_json(args.preparation), read_json(args.baseline)
    with output_lease(output), device_lease(args.device, '/root/autodl-tmp/.b4dl_gpu0_eval.lock'):
        tokenizer, original = load_original(lock, args.device)
        if args.command == 'verify':
            chosen = {}
            for row in rows: chosen.setdefault(row['task'], row)
            expected = {}
            original.eval()
            for task, row in chosen.items():
                batch = query_batch(original, tokenizer, dataset, row)
                expected[task] = original.generate(**batch, max_new_tokens=8, do_sample=False,
                    use_cache=True, pad_token_id=tokenizer.eos_token_id)
            model, options = attach(original, 'A3', chosen_config(preparation, 'A3'), 0)
            model.eval()
            for task, row in chosen.items():
                actual = model.generate(**query_batch(model, tokenizer, dataset, row), max_new_tokens=8,
                    do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                if not torch.equal(actual, expected[task]): raise ValueError('six-task zero-init tokens changed: '+task)
            write_json(output/'verification.json', {'status': 'six_task_adapter_runtime_verified',
                'tasks': list(chosen), 'request_ids': [r['request_id'] for r in chosen.values()],
                'all_zero_init_tokens_exact': True, 'generation_budget': 8, 'optimizer_steps': 0,
                'engineering_only': True, 'performance_measured': False, 'protocol_id': protocol['protocol_id'],
                'identity': identity_for(args, dataset)})
            print('six_task_adapter_runtime_verified', flush=True)
            return
        plan = read_json(args.experiment) if args.experiment else None
        rank = plan['recipe']['lora_rank'] if plan else 0
        model, options = attach(original, args.arm, chosen_config(preparation, args.arm), rank if args.arm != 'A0' else 0)
        identity = identity_for(args, dataset)
        if args.checkpoint: load_checkpoint(args.checkpoint, model, options, identity)
        model.eval()
        run = {'schema': 'b4dl-core-six-task-inference-v2',
            'identity': identity, 'protocol_id': protocol['protocol_id'], 'arm': args.arm,
            'checkpoint_sha256': sha256(args.checkpoint) if args.checkpoint else sha256(args.baseline),
            'generation': protocol['generation'], 'oracle_assisted_metadata': True,
            'options': options, 'preparation_sha256': digest(preparation),
            'requests_sha256': digest(rows), 'device': args.device,
            'runtime': runtime_identity(language_model(model)),
            'dtype': str(language_model(model).get_input_embeddings().weight.dtype)}
        infer_predictions(output, run, rows, dataset, model, tokenizer, args.resume)


if __name__ == '__main__': main()
