"""Freeze reproducible training and six-task regression inputs before SFT."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from .inputs import GeometryDataset
from .integration import prompt_tokens
from .io import digest, read_json, sha256, write_json, write_jsonl


def validate_experiment(args, preparation, lock, dataset):
    path = getattr(args, "experiment", None)
    if not path or preparation.get("experiment_sha256") != sha256(path):
        raise ValueError("formal training requires the frozen experiment manifest")
    plan = read_json(path)
    if plan["schema"] != "b4dl-core-experiment-v1" or plan["experiment_id"] != digest({k: v for k, v in plan.items() if k != "experiment_id"}):
        raise ValueError("experiment identity mismatch")
    if plan["dataset_id"] != dataset.manifest["dataset_id"] or plan["baseline_manifest_sha256"] != sha256(args.baseline):
        raise ValueError("experiment data/baseline mismatch")
    recipe = plan["recipe"]
    for key in ("lr", "batch_size", "grad_accum", "steps", "lora_rank", "save_every", "keep_last"):
        if getattr(args, key, None) != recipe[key]:
            raise ValueError("command disagrees with frozen recipe: " + key)
    if args.arm not in plan["trainable_arms"] or args.seed not in plan["seeds"]:
        raise ValueError("unregistered arm/seed")
    if len(dataset.requests) != recipe["train_examples"] or recipe["steps"]*recipe["batch_size"]*recipe["grad_accum"] != recipe["train_examples"]*recipe["epochs"]:
        raise ValueError("frozen training exposure count mismatch")
    protocol = plan["six_task_protocol"]
    if sha256(protocol["path"]) != protocol["sha256"]:
        raise ValueError("six-task protocol changed")
    frozen = read_json(protocol["path"])
    if frozen["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("six-task protocol identity mismatch")
    for name, expected in frozen["files_sha256"].items():
        if sha256(name) != expected:
            raise ValueError("six-task input/source changed: " + name)
    return plan


def prepare_protocol(project_root, dataset_root, baseline, output, nltk_root):
    import os
    os.environ['NLTK_DATA'] = str(nltk_root)
    import nltk
    nltk.data.path.insert(0, str(nltk_root))
    from transformers import AutoTokenizer
    from evaluation.test_b4dl import B4DLEvaluator, build_query
    from vtimellm.conversation import conv_templates
    from vtimellm.mm_utils import tokenizer_image_token
    from vtimellm.constants import IMAGE_TOKEN_INDEX
    project_root, output = Path(project_root), Path(output)
    if output.exists(): raise FileExistsError("choose a new protocol directory")
    output.mkdir(parents=True)
    dataset = GeometryDataset(dataset_root)
    lock = read_json(baseline)
    material_root = project_root/'mllm/b4dl_dataset'
    items = read_json(material_root/'test_qa.json')
    if isinstance(items, dict): items = list(items.values())
    ego, motion = read_json(material_root/'ego_metadata.json'), read_json(material_root/'ego_frame_motion.json')
    rows, counts = [], Counter()
    for index, item in enumerate(items):
        task = B4DLEvaluator.canonical_task(item.get('task') or '')
        if task not in B4DLEvaluator.ALL_TASKS or len(item.get('conversations', [])) != 2:
            raise ValueError("malformed/unknown six-task item")
        ident = item.get('scene_id') or item.get('id')
        if dataset.manifest['scenes'][ident]['split'] != 'test':
            raise ValueError("six-task QA not in official validation scenes")
        target = item['conversations'][1]['value']
        query = build_query(item['conversations'][0]['value'], ego_meta=ego, scene_id=ident,
            use_4dlidar=True, use_meta=True, per_sequence=True, frame_motion=motion,
            gt=target, answer_frames=True, task=item['task'])
        scene = dataset.manifest['scenes'][ident]
        fingerprint = digest({'query': query, 'scene_bundle_sha256': scene['sha256'], 'whole_scene': True})
        rows.append({'request_id': digest([index, task, ident, fingerprint])[:24], 'source_index': index,
            'task': task, 'scene_id': ident, 'query': query, 'target': target, 'input_fingerprint': fingerprint})
        counts[task] += 1
    if len(rows) != 30145 or set(counts) != set(B4DLEvaluator.ALL_TASKS):
        raise ValueError("six-task regression denominator is incomplete")
    tokenizer = AutoTokenizer.from_pretrained(lock['components']['model_base']['path'], use_fast=False, local_files_only=True)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<4DLiDAR>', '<meta>']})
    config = read_json(Path(lock['components']['model_base']['path'])/'config.json')
    context_limit = min(v for v in (config.get('max_position_embeddings'), config.get('tokenizer_model_max_length')) if v)
    max_six, max_training = 0, 0
    for row in rows:
        conv = conv_templates['v1'].copy()
        conv.append_message(conv.roles[0], row['query']); conv.append_message(conv.roles[1], None)
        length = len(tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX))-1+dataset.manifest['scenes'][row['scene_id']]['frames']
        max_six = max(max_six, length)
    print('six-task text/visual context checked', len(rows), flush=True)
    for n, request in enumerate(dataset.requests):
        if request['split'] != 'train': continue
        target = dataset.targets[request['request_id']]
        answer = json.dumps({'answer': target['answer'], 'evidence': target['evidence']}, ensure_ascii=False, allow_nan=False)
        length = len(prompt_tokens(tokenizer, request['prompt']))-1+request['feature_length']+len(tokenizer.encode(answer, add_special_tokens=False))+1
        max_training = max(max_training, length)
        if n % 10000 == 0: print('training context checked through source row', n, flush=True)
    if max_six+1024 > context_limit or max_training > context_limit:
        raise ValueError('expanded context exceeds model limit')
    request_path = output/'six-task-requests.jsonl'
    write_jsonl(request_path, rows)
    files = [material_root/name for name in ('test_qa.json', 'ego_metadata.json', 'ego_frame_motion.json')]
    backends = {name: Path(sys.modules[module].__file__) for name, module in
        (('query_builder', build_query.__module__), ('evaluator', B4DLEvaluator.__module__), ('ego_renderer', 'ego_text'))}
    files += list(backends.values()) + [request_path]
    bert = project_root/'models/roberta-large'
    for name in ('config.json', 'model.safetensors', 'vocab.json', 'merges.txt', 'tokenizer.json', 'tokenizer_config.json'):
        files.append(bert/name)
    files += [p for p in Path(nltk_root).rglob('*') if p.is_file()]
    protocol = {'schema': 'b4dl-core-six-task-v1', 'dataset_id': dataset.manifest['dataset_id'],
        'request_path': str(request_path), 'requests': len(rows), 'task_counts': dict(counts),
        'input_protocol': {'whole_scene': True, 'per_sequence_metadata': True, 'answer_frames': True,
            'oracle_assisted_metadata': True, 'prompt_template': 'v1', 'add_geometry_instruction': False},
        'generation': {'max_new_tokens': 1024, 'do_sample': False, 'num_beams': 1, 'use_cache': True},
        'metrics': {'meteor_backend': 'dual', 'primary_meteor': 'nltk-2005', 'bert_model_path': str(bert),
            'nltk_data': str(nltk_root), 'bertscore_layer': 17, 'gpt_score_enabled': False, 'errors_retained_in_denominator': True},
        'context_audit': {'model_limit': context_limit, 'maximum_six_task_prompt_and_visual': max_six,
            'maximum_training_expanded_length': max_training, 'six_task_generation_budget': 1024},
        'files_sha256': {str(p): sha256(p) for p in files},
        'backend_sources': {name: {'path': str(p), 'sha256': sha256(p)} for name, p in backends.items()},
        'selection': 'fixed final endpoint; no test-based checkpoint selection; same inputs for A0 and extensions'}
    protocol['protocol_id'] = digest(protocol)
    write_json(output/'six-task-manifest.json', protocol)
    print(protocol['protocol_id'], counts, protocol['context_audit'], flush=True)
    return protocol


def seal_java_runtime(protocol_path, java_home, output):
    protocol = read_json(protocol_path)
    if protocol['protocol_id'] != digest({k: v for k, v in protocol.items() if k != 'protocol_id'}):
        raise ValueError('parent protocol changed')
    if Path(output).exists(): raise FileExistsError('use a new protocol output')
    java_home = Path(java_home).resolve()
    if not (java_home/'bin/java').is_file(): raise ValueError('missing Java runtime')
    protocol.pop('protocol_id')
    protocol['parent_protocol_sha256'] = sha256(protocol_path)
    protocol['metrics']['java_home'] = str(java_home)
    for path in java_home.rglob('*'):
        if path.is_file(): protocol['files_sha256'][str(path)] = sha256(path)
    protocol['protocol_id'] = digest(protocol)
    write_json(output, protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'dataset', 'baseline', 'output', 'nltk-root'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    prepare_protocol(args.project_root, args.dataset, args.baseline, args.output, args.nltk_root)


if __name__ == '__main__': main()
