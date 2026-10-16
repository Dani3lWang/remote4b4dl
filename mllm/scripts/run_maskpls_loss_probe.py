#!/usr/bin/env python
"""Bounded, paired loss-policy screen using only an audited parent's training scenes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from grounded4d.metrics import match_sets
from grounded4d.train import to_device
from maskpls_b4dl.config import MaskPLSConfig
from maskpls_b4dl.criterion import MaskPLSCriterion
from maskpls_b4dl.data import MaskPLSDataset
from maskpls_b4dl.model import MaskPLS, extract_instances
from maskpls_b4dl.protocol import implementation_digest
from maskpls_b4dl.runner import check_warm_start_data, load_warm_start, make_dataset


def hashed(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def log(path, data):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)


def bucket(n):
    return '1-10' if n <= 10 else '11-49' if n < 50 else '50+'


def summary(rows):
    totals = {key: sum(r[key] for r in rows) for key in
              ('gt', 'pred', 'tp', 'raw_geometry_tp', 'raw_class_tp', 'predicted_points', 'ignored_points', 'ego_points')}
    totals.update(frames=len(rows), precision=totals['tp'] / max(1, totals['pred']),
                  recall=totals['tp'] / max(1, totals['gt']),
                  raw_geometry_recall=totals['raw_geometry_tp'] / max(1, totals['gt']),
                  ignored_fraction=totals['ignored_points'] / max(1, totals['predicted_points']))
    totals['point_buckets'] = {}
    for name in ('1-10', '11-49', '50+'):
        values = {key: sum(r['point_buckets'][name][key] for r in rows)
                  for key in ('gt', 'raw_geometry_tp', 'raw_class_tp', 'tp')}
        values['raw_geometry_recall'] = values['raw_geometry_tp'] / values['gt'] if values['gt'] else None
        values['recall'] = values['tp'] / values['gt'] if values['gt'] else None
        totals['point_buckets'][name] = values
    return totals


def matrix_iou(gt, masks):
    inter = gt.float() @ masks.float().T
    union = gt.sum(1)[:, None] + masks.sum(1)[None] - inter
    return torch.where(union > 0, inter.double() / union, 0).cpu().numpy()


@torch.no_grad()
def evaluate(model, dataset, indices, output, stage, progress_path):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'predictions').mkdir()
    model.eval()
    rows = []
    for ordinal, index in enumerate(indices):
        cpu_frame = dataset[index]['frames'][0]
        frame = to_device(cpu_frame, torch.device('cuda:0'))
        prediction = model(frame['points'])
        objects = extract_instances(prediction, model.config)
        valid = prediction['valid_points'] & frame['semantic_labels'].ne(0)
        gt = frame['labels'].eq(1)[:, valid]
        raw = prediction['mask_logits'].sigmoid().ge(model.config.mask_threshold)[:, valid]
        final = objects['masks'][:, valid]
        gt_class = frame['class_ids'].cpu().numpy()
        query_class = objects['all_classes'].cpu().numpy() - 1
        raw_iou, final_iou = matrix_iou(gt, raw), matrix_iou(gt, final)
        _, geometry = match_sets(raw_iou, np.zeros(len(gt)), np.zeros(len(raw)))
        _, semantic = match_sets(raw_iou, gt_class, query_class)
        _, matches = match_sets(final_iou, gt_class, objects['class_ids'].cpu().numpy())
        hits = {name: {int(p[0]) for p in pairs} for name, pairs in
                [('raw_geometry_tp', geometry), ('raw_class_tp', semantic), ('tp', matches)]}
        sizes = gt.sum(1).cpu().numpy()
        strata = {name: dict(gt=0, raw_geometry_tp=0, raw_class_tp=0, tp=0) for name in ('1-10','11-49','50+')}
        for t, size in enumerate(sizes):
            current = strata[bucket(size)]; current['gt'] += 1
            for key in hits:
                current[key] += int(t in hits[key])
        source = dataset.records[index]['frames'][0]
        with np.load(dataset.panoptic_root / source['source_panoptic_file']['path'], allow_pickle=False) as z:
            pan = np.asarray(z['data']).reshape(-1)
        masks = objects['masks'].cpu().numpy()
        row = dict(dataset_index=index, token=frame['sample_data_token'], scene_token=frame['scene_token'],
                   gt=len(gt), pred=len(final), tp=len(matches), raw_geometry_tp=len(geometry), raw_class_tp=len(semantic),
                   point_buckets=strata, predicted_points=int(masks.sum()),
                   ignored_points=int(masks[:, (prediction['valid_points'] & ~valid).cpu().numpy()].sum()),
                   ego_points=int(masks[:, pan // 1000 == 31].sum()),
                   point_file=source['point_file'], mask_file=source['mask_file'],
                   panoptic_file=source['source_panoptic_file'])
        snapshot = output / 'predictions' / (row['token'] + '.npz')
        np.savez_compressed(snapshot, valid=valid.cpu().numpy(),
                            roi=prediction['valid_points'].cpu().numpy(),
                            full_points=np.asarray(len(frame['points'])),
                            final_full_packed=np.packbits(masks, axis=1, bitorder='little'),
                            raw_packed=np.packbits(raw.cpu().numpy(), axis=1, bitorder='little'),
                            final_packed=np.packbits(final.cpu().numpy(), axis=1, bitorder='little'),
                            metric_points=np.asarray(int(valid.sum())), gt_class=gt_class, query_class=query_class,
                            final_class=objects['class_ids'].cpu().numpy())
        row['snapshot_sha256'] = hashed(snapshot)
        log(output / 'frames.jsonl', row)
        rows.append(row)
        write_json(progress_path, dict(stage=stage, completed_frames=ordinal + 1, expected_frames=len(indices)))
    totals = summary(rows)
    write_json(output / 'metrics.json', totals)
    return totals, rows


def load_model(state, config, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = MaskPLS(config).to('cuda:0')
    model.load_state_dict(state['model'], strict=True)
    return model


def run(args):
    if not 1 <= args.steps <= 320:
        raise ValueError('loss screen requires a bounded 1..320 optimizer-update budget')
    if hashed(args.checkpoint) != args.source_weights_sha256:
        raise ValueError('parent weights hash mismatch')
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = MaskPLSConfig.from_dict(source['metadata']['config'])
    if config.mask_ignore_policy != 'background':
        raise ValueError('this screen preserves corrected background negative supervision')
    if config.mask_bce_reduction != 'point_mean':
        raise ValueError('parent baseline must use point_mean')
    source = load_warm_start(args.checkpoint, config, args.source_implementation_sha256)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    train, train_report = make_dataset(args, config)
    dev, dev_report = make_dataset(args, config, development=True)
    check_warm_start_data(source['metadata'], dict(data=train_report, internal_dev_data=dev_report))
    if train.scenes & dev.scenes:
        raise ValueError('train/dev scenes overlap')
    grouped = {}
    for index, row in enumerate(train.records):
        grouped.setdefault(row['scene_token'], []).append(index)
    probe = []
    for scene in sorted(grouped):
        order = sorted(grouped[scene], key=lambda i:train.records[i]['frames'][0]['timestamp_us'])
        probe.append(order[len(order)//2])
    fit = [probe[i] for i in np.linspace(0, len(probe)-1, min(16,len(probe)), dtype=int)]
    dev_indices = list(range(len(dev)))
    sequence = []
    generator = random.Random(args.seed)
    while len(sequence) < args.steps:
        cycle = fit.copy(); generator.shuffle(cycle); sequence.extend(cycle)
    sequence = sequence[:args.steps]
    plan = dict(scope='single-seed bounded loss-policy screen, not final model acceptance',
                source_weights_sha256=args.source_weights_sha256,
                source_implementation_sha256=args.source_implementation_sha256,
                implementation_sha256=implementation_digest(), script_sha256=hashed(__file__),
                config=config.to_dict(),
                ignored_raw_semantic_ids=train_report['class_admission']['ignored_raw_semantic_ids'],
                parent_completed_epoch=source['completed_epoch'], parent_global_step=source['global_step'],
                train_manifest_sha256=train.manifest_sha256, dev_manifest_sha256=dev.manifest_sha256,
                train_probe_indices=probe, fit_indices=fit, dev_indices=dev_indices,
                fit_frame_tokens=[train.records[i]['frames'][0]['sample_data_token'] for i in fit],
                optimization_sequence=sequence, seed=args.seed, steps=args.steps, batch_size=1, accumulate=1,
                lr=args.lr, gradient_clip=1., weight_decay=.01, scheduler='constant, identical in both arms',
                source_data_identity_verified=True, subset_role='subset of parent train only; not a production warm-start with changed data',
                changed_variable='mask_bce_reduction only: point_mean vs instance_balanced',
                fixed='matching cost, Dice, classification, semantic loss, sampling, architecture, ROI and inference thresholds',
                preregistered_screen=dict(primary='B dev 1-10 point raw geometry TP >= max(A, frozen)+5',
                  final_tp='B dev final TP >= max(A, frozen)', precision='B dev precision >= frozen precision - 0.05',
                  ignored_fraction='B dev ignored fraction <= max(A, frozen)+0.02',
                  ego_points='B dev ego points <= max(A, frozen)*1.1+100',
                  caveat='screening margins are engineering choices, not statistical confidence intervals'),
                model_performance_accepted=False)
    write_json(args.output_dir / 'plan.json', plan)
    torch.set_float32_matmul_precision('highest')
    model = load_model(source, config, args.seed)
    frozen_train, frozen_rows = evaluate(model, train, probe, args.output_dir / 'frozen_train_probe',
                                         'frozen_train_probe', args.output_dir / 'progress.json')
    frozen_dev, _ = evaluate(model, dev, dev_indices, args.output_dir / 'frozen_dev',
                             'frozen_dev', args.output_dir / 'progress.json')
    frozen_fit = summary([r for r in frozen_rows if r['dataset_index'] in set(fit)])
    write_json(args.output_dir / 'frozen_fit_metrics.json', frozen_fit)
    del model; torch.cuda.empty_cache()
    arms = {}
    for arm, reduction in [('A', 'point_mean'), ('B', 'instance_balanced')]:
        arm_dir = args.output_dir / arm; arm_dir.mkdir()
        current = MaskPLSConfig.from_dict(dict(config.to_dict(), mask_bce_reduction=reduction))
        model = load_model(source, current, args.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
        criterion = MaskPLSCriterion(current)
        model.train()
        for step, index in enumerate(sequence, 1):
            # Reset per-update RNG: identical dropout and target sampling streams.
            torch.manual_seed(args.seed + step); torch.cuda.manual_seed_all(args.seed + step)
            frame = to_device(train[index]['frames'][0], torch.device('cuda:0'))
            optimizer.zero_grad(set_to_none=True)
            loss, parts, _ = criterion(model(frame['points']), frame)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            log(arm_dir / 'steps.jsonl', dict(step=step, dataset_index=index, loss=float(loss.detach()),
                gradient_norm=float(norm), **{k:float(v.detach()) for k,v in parts.items()}))
            write_json(args.output_dir / 'progress.json', dict(stage='train_' + arm, completed_updates=step, expected_updates=args.steps))
            if step in (args.steps//2, args.steps):
                # Deliberately distinct schema: production resume must not mistake a subset probe for full-data training.
                torch.save(dict(schema_version='b4dl-maskpls-loss-probe-v1', config=current.to_dict(),
                    model=model.state_dict(), optimizer=optimizer.state_dict(), completed_updates=step,
                    source_weights_sha256=args.source_weights_sha256, fit_frame_tokens=plan['fit_frame_tokens'],
                    implementation_sha256=implementation_digest(), model_performance_accepted=False),
                    arm_dir / ('step-%04d.pt' % step))
        fit_metrics, _ = evaluate(model, train, fit, arm_dir / 'fit_eval', 'fit_eval_' + arm, args.output_dir / 'progress.json')
        dev_metrics, _ = evaluate(model, dev, dev_indices, arm_dir / 'dev_eval', 'dev_eval_' + arm, args.output_dir / 'progress.json')
        arms[arm] = dict(fit=fit_metrics, dev=dev_metrics,
                         checkpoint_sha256=hashed(arm_dir / ('step-%04d.pt' % args.steps)))
        del model, optimizer, criterion, loss, parts, frame; torch.cuda.empty_cache()
    a, b = arms['A']['dev'], arms['B']['dev']
    gates = dict(small_geometry=b['point_buckets']['1-10']['raw_geometry_tp'] >=
                     max(a['point_buckets']['1-10']['raw_geometry_tp'],frozen_dev['point_buckets']['1-10']['raw_geometry_tp'])+5,
                 final_tp=b['tp']>=max(a['tp'],frozen_dev['tp']),
                 precision=b['precision']>=frozen_dev['precision']-.05,
                 ignored_fraction=b['ignored_fraction']<=max(a['ignored_fraction'],frozen_dev['ignored_fraction'])+.02,
                 ego_points=b['ego_points']<=max(a['ego_points'],frozen_dev['ego_points'])*1.1+100)
    result = dict(frozen_train_probe=frozen_train, frozen_fit=frozen_fit, frozen_dev=frozen_dev, arms=arms,
                  gates=gates, screen_passed=all(gates.values()), model_performance_accepted=False,
                  source_weights_unchanged=hashed(args.checkpoint)==args.source_weights_sha256)
    if not result['source_weights_unchanged']:
        raise ValueError('parent checkpoint changed during screen')
    write_json(args.output_dir / 'comparison.json', result)
    write_json(args.output_dir / 'inference_training_completion.json', dict(complete=True, steps_per_arm=args.steps,
                all_evaluation_frames_completed=True, source_weights_unchanged=True, screen_passed=result['screen_passed']))
    print('PAIRED_LOSS_SCREEN_COMPLETED', flush=True)


def independent_pairs(iou, gc, pc):
    n, m = iou.shape
    eligible = (iou >= .5) & (gc[:, None] == pc[None])
    objective = np.zeros((n, m+n)); objective[:, :m] = eligible * (1+iou/(n+1))
    r,c = linear_sum_assignment(-objective)
    return {int(i) for i,j in zip(r,c) if j<m and eligible[i,j]}


def continue_screen(args):
    """One additional prespecified arm; reuse audited A/frozen references without retraining them."""
    if args.reference_dir is None:
        raise ValueError('continuation requires an audited reference directory')
    parent_plan=json.loads((args.reference_dir/'plan.json').read_text())
    previous=json.loads((args.reference_dir/'comparison.json').read_text())
    previous_audit=json.loads((args.reference_dir/'audit.json').read_text())
    if not previous_audit['complete'] or args.steps!=parent_plan['steps'] or args.lr!=parent_plan['lr'] or args.seed!=parent_plan['seed']:
        raise ValueError('continuation must retain the audited budget and random stream')
    if hashed(args.checkpoint)!=parent_plan['source_weights_sha256'] or hashed(args.checkpoint)!=args.source_weights_sha256:
        raise ValueError('continuation parent weights differ')
    parent=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=MaskPLSConfig.from_dict(dict(parent['metadata']['config'],mask_bce_reduction=args.policy))
    parent=load_warm_start(args.checkpoint,config,args.source_implementation_sha256)
    train,tr=make_dataset(args,config);dev,dr=make_dataset(args,config,development=True)
    check_warm_start_data(parent['metadata'],dict(data=tr,internal_dev_data=dr))
    if train.manifest_sha256!=parent_plan['train_manifest_sha256'] or dev.manifest_sha256!=parent_plan['dev_manifest_sha256'] or train.scenes&dev.scenes:
        raise ValueError('continuation data identity changed')
    fit=parent_plan['fit_indices'];sequence=parent_plan['optimization_sequence']
    assert [train.records[i]['frames'][0]['sample_data_token'] for i in fit]==parent_plan['fit_frame_tokens']
    args.output_dir.mkdir(parents=True,exist_ok=False)
    plan=dict(parent_plan,implementation_sha256=implementation_digest(),script_sha256=hashed(__file__),
              new_arm='C',new_policy=args.policy,
              reference_dir=str(args.reference_dir),reference_comparison_sha256=hashed(args.reference_dir/'comparison.json'),
              reference_audit_sha256=hashed(args.reference_dir/'audit.json'),
              reason='full 50/50 balancing broadened masks; cap positive reweighting without removing negative supervision')
    write_json(args.output_dir/'plan.json',plan)
    arm_dir=args.output_dir/'C';arm_dir.mkdir()
    torch.set_float32_matmul_precision('highest')
    model=load_model(parent,config,args.seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    criterion=MaskPLSCriterion(config);model.train()
    for step,index in enumerate(sequence,1):
        torch.manual_seed(args.seed+step);torch.cuda.manual_seed_all(args.seed+step)
        frame=to_device(train[index]['frames'][0],torch.device('cuda:0'))
        optimizer.zero_grad(set_to_none=True)
        loss,parts,_=criterion(model(frame['points']),frame)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite continuation loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        log(arm_dir/'steps.jsonl',dict(step=step,dataset_index=index,loss=float(loss.detach()),gradient_norm=float(norm),
                                    **{k:float(v.detach()) for k,v in parts.items()}))
        write_json(args.output_dir/'progress.json',dict(stage='train_C',completed_updates=step,expected_updates=args.steps))
        if step in (args.steps//2,args.steps):
            torch.save(dict(schema_version='b4dl-maskpls-loss-probe-v1',config=config.to_dict(),model=model.state_dict(),
                optimizer=optimizer.state_dict(),completed_updates=step,source_weights_sha256=args.source_weights_sha256,
                fit_frame_tokens=plan['fit_frame_tokens'],implementation_sha256=implementation_digest(),model_performance_accepted=False),
                arm_dir/('step-%04d.pt'%step))
    fit_metrics,_=evaluate(model,train,fit,arm_dir/'fit_eval','fit_eval_C',args.output_dir/'progress.json')
    dev_metrics,_=evaluate(model,dev,parent_plan['dev_indices'],arm_dir/'dev_eval','dev_eval_C',args.output_dir/'progress.json')
    a=previous['arms']['A']['dev'];frozen=previous['frozen_dev'];b=dev_metrics
    gates=dict(small_geometry=b['point_buckets']['1-10']['raw_geometry_tp']>=max(a['point_buckets']['1-10']['raw_geometry_tp'],frozen['point_buckets']['1-10']['raw_geometry_tp'])+5,
               final_tp=b['tp']>=max(a['tp'],frozen['tp']),precision=b['precision']>=frozen['precision']-.05,
               ignored_fraction=b['ignored_fraction']<=max(a['ignored_fraction'],frozen['ignored_fraction'])+.02,
               ego_points=b['ego_points']<=max(a['ego_points'],frozen['ego_points'])*1.1+100)
    current=dict(previous,new_arm='C',previous_gates=previous['gates'],gates=gates,screen_passed=all(gates.values()),
                 source_weights_unchanged=hashed(args.checkpoint)==args.source_weights_sha256)
    current['arms']=dict(previous['arms'],C=dict(fit=fit_metrics,dev=dev_metrics,
                           checkpoint_sha256=hashed(arm_dir/('step-%04d.pt'%args.steps))))
    assert current['source_weights_unchanged']
    write_json(args.output_dir/'comparison.json',current)
    write_json(args.output_dir/'inference_training_completion.json',dict(complete=True,new_arm='C',steps_per_arm=args.steps,
               all_evaluation_frames_completed=True,source_weights_unchanged=True,screen_passed=current['screen_passed']))
    print('CAPPED_POSITIVE_LOSS_SCREEN_COMPLETED',flush=True)


def cpu_iou(a, b):
    intersection = a.astype(np.float32) @ b.astype(np.float32).T
    union = a.sum(1)[:,None] + b.sum(1)[None] - intersection
    return np.divide(intersection,union,out=np.zeros(intersection.shape,dtype=np.float64),where=union>0)


def audit(args):
    plan = json.loads((args.output_dir / 'plan.json').read_text())
    complete = json.loads((args.output_dir / 'inference_training_completion.json').read_text())
    assert complete['complete'] and hashed(args.checkpoint)==plan['source_weights_sha256']
    assert hashed(args.manifest)==plan['train_manifest_sha256']
    assert hashed(args.validation_manifest)==plan['dev_manifest_sha256']
    if args.audit_arm:
        if args.audit_arm!='C' or args.reference_dir is None:raise ValueError('audit continuation requires C reference')
        assert hashed(args.reference_dir/'audit.json')==plan['reference_audit_sha256']
        assert hashed(args.reference_dir/'comparison.json')==plan['reference_comparison_sha256']
        scopes=['C/fit_eval','C/dev_eval']
    else:
        scopes = ['frozen_train_probe','frozen_dev','A/fit_eval','A/dev_eval','B/fit_eval','B/dev_eval']
    audited = {}
    for scope in scopes:
        directory = args.output_dir / scope
        rows = [json.loads(line) for line in (directory / 'frames.jsonl').read_text().splitlines()]
        assert len(rows)==(len(plan['train_probe_indices']) if scope=='frozen_train_probe' else len(plan['fit_indices']) if 'fit_eval' in scope else len(plan['dev_indices']))
        assert len({r['token'] for r in rows})==len(rows)
        for row in rows:
            path=directory/'predictions'/(row['token']+'.npz')
            assert hashed(path)==row['snapshot_sha256']
            assert hashed(args.label_root/row['mask_file']['path'])==row['mask_file']['sha256']
            assert hashed(args.nuscenes_root/row['point_file']['path'])==row['point_file']['sha256']
            assert hashed(args.panoptic_root/row['panoptic_file']['path'])==row['panoptic_file']['sha256']
            with np.load(args.label_root/row['mask_file']['path'],allow_pickle=False) as z:labels=z['labels']==1
            xyz=np.fromfile(args.nuscenes_root/row['point_file']['path'],dtype='<f4').reshape(-1,5)[:,:3]
            with np.load(args.panoptic_root/row['panoptic_file']['path'],allow_pickle=False) as z:pan=np.asarray(z['data']).reshape(-1)
            bounds=np.asarray(plan['config']['point_cloud_range'])
            roi=((xyz>=bounds[:3])&(xyz<bounds[3:])).all(1)
            expected_valid=roi&~np.isin(pan//1000,plan['ignored_raw_semantic_ids'])
            with np.load(path,allow_pickle=False) as z:
                assert np.array_equal(z['roi'],roi) and np.array_equal(z['valid'],expected_valid)
                gt=labels[:,z['valid']];gc=z['gt_class'];count=int(z['metric_points'])
                raw=np.unpackbits(z['raw_packed'],axis=1,count=count,bitorder='little').astype(bool)
                final=np.unpackbits(z['final_packed'],axis=1,count=count,bitorder='little').astype(bool)
                full=np.unpackbits(z['final_full_packed'],axis=1,count=int(z['full_points']),bitorder='little').astype(bool)
                assert np.array_equal(full[:,expected_valid],final) and not full[:,~roi].any()
                assert row['predicted_points']==int(full.sum())
                assert row['ignored_points']==int(full[:,roi&~expected_valid].sum())
                assert row['ego_points']==int(full[:,pan//1000==31].sum())
                ri,fi=cpu_iou(gt,raw),cpu_iou(gt,final)
                groups={'raw_geometry_tp':independent_pairs(ri,np.zeros(len(gc)),np.zeros(len(raw))),
                        'raw_class_tp':independent_pairs(ri,gc,z['query_class']),
                        'tp':independent_pairs(fi,gc,z['final_class'])}
                assert row['gt']==len(gt) and row['pred']==len(final)
                for key,hits in groups.items():assert row[key]==len(hits),(scope,row['token'],key)
                strata={name:dict(gt=0,raw_geometry_tp=0,raw_class_tp=0,tp=0) for name in ('1-10','11-49','50+')}
                for i,size in enumerate(gt.sum(1)):
                    item=strata[bucket(size)];item['gt']+=1
                    for key,hits in groups.items():item[key]+=int(i in hits)
                assert strata==row['point_buckets']
        recomputed=summary(rows)
        assert recomputed==json.loads((directory/'metrics.json').read_text())
        audited[scope]=dict(frames=len(rows),gt=recomputed['gt'],tp=recomputed['tp'])
        print(json.dumps({'audited':scope,**audited[scope]}),flush=True)
    comparison=json.loads((args.output_dir/'comparison.json').read_text())
    for arm in ((args.audit_arm,) if args.audit_arm else ('A','B')):
        ckpt=args.output_dir/arm/('step-%04d.pt'%args.steps)
        assert hashed(ckpt)==comparison['arms'][arm]['checkpoint_sha256']
        for scope in ('fit','dev'):
            assert comparison['arms'][arm][scope]==json.loads((args.output_dir/arm/(scope+'_eval')/'metrics.json').read_text())
    write_json(args.output_dir/'audit.json',dict(complete=True,independent_numpy_iou=True,
        independent_scipy_matching=True,all_point_buckets_verified=True,scopes=audited,
        source_weights_unchanged=True,model_performance_accepted=False))
    print('INDEPENDENT_LOSS_SCREEN_AUDIT_COMPLETED',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','audit','continuation'])
    parser.add_argument('--reference-dir',type=Path)
    parser.add_argument('--policy',choices=['sqrt_positive'],default='sqrt_positive')
    parser.add_argument('--audit-arm',choices=['C'])
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source-weights-sha256',required=True)
    parser.add_argument('--source-implementation-sha256',required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--validation-manifest',type=Path,required=True)
    parser.add_argument('--scene-tokens',type=Path,required=True)
    parser.add_argument('--dev-scene-tokens',type=Path,required=True)
    parser.add_argument('--nuscenes-root',type=Path,required=True)
    parser.add_argument('--metadata-root',type=Path,required=True)
    parser.add_argument('--panoptic-root',type=Path,required=True)
    parser.add_argument('--label-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=160)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args();args.command='evaluate';args.split='train';args.official_val=False
    if args.mode=='run':run(args)
    elif args.mode=='continuation':continue_screen(args)
    else:audit(args)
