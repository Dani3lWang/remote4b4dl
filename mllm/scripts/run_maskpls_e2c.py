"""Paired E2-C: audited epoch14 migration, fixed samples, two 3-epoch arms.

Run with the isolated wqlc environment. This is a new controlled experiment,
not an exact continuation of the parent's optimizer or a new holdout score.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from grounded4d.data import load_scene_tokens
from grounded4d.metrics import match_sets
from grounded4d.train import to_device, rng_state
from maskpls_b4dl.config import MaskPLSConfig
from maskpls_b4dl.criterion import MaskPLSCriterion, sample_supervised_points
from maskpls_b4dl.data import MaskPLSDataset, build_targets
from maskpls_b4dl.model import MaskPLS, extract_instances
from maskpls_b4dl.protocol import implementation_digest
from maskpls_b4dl.runner import base_metadata, load_warm_start, load_native

PARENT_SHA='b6b1fd0377ac206304eea9b4b944a0df92564cd93d1a9976e0c8b67d8cf5665b'
PARENT_IMPLEMENTATION='22d7c7ecce647817f41b8371e850551fb4e6831940a4b5090d2f4845c17cfb16'
ARMS={'C0':.1,'C1':.05}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def log(path,value):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def array(t):return t.detach().cpu().numpy().copy()


def digest(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str((t.dtype,tuple(t.shape))).encode());h.update(array(t).tobytes())
    return h.hexdigest()


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
    torch.set_float32_matmul_precision('highest')


def bucket(n):return '1-2' if n<=2 else '3-10' if n<=10 else '11-49' if n<50 else '50+'


def pairs(iou,gc,pc,geometry=False):
    allowed=iou>=.5
    if not geometry:allowed &= np.asarray(gc)[:,None]==np.asarray(pc)[None]
    n,m=iou.shape;score=np.zeros((n,m+n))
    score[:,:m]=allowed*(1+iou/(n+1))
    rows,cols=linear_sum_assignment(-score)
    return [(int(i),int(j)) for i,j in zip(rows,cols) if j<m and allowed[i,j]]


def cpu_iou(gt,pred):
    inter=np.stack([pred[:,np.flatnonzero(g)].sum(1) for g in gt]) if len(gt) else np.zeros((0,len(pred)))
    union=gt.sum(1)[:,None]+pred.sum(1)[None]-inter
    return np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0)


def gpu_iou(gt,pred):
    inter=array(gt.float()@pred.float().T).astype(float)
    union=array(gt.sum(1))[:,None]+array(pred.sum(1))[None]-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)


def datasets(a,cfg):
    tv=a.parent_root/'datageneration/data/trainval-64-001'
    result={}
    for scope,manifest,scenes in [('train_probe','train','train'),('dev','internal_dev','dev')]:
        result[scope]=MaskPLSDataset(tv/(manifest+'.jsonl'),a.data,tv,cfg.data_config(),
            load_scene_tokens(tv/(scenes+'_scene_tokens.json')),split=manifest,supervision='panoptic',
            panoptic_root=a.data/'nuScenes-panoptic-v1.0-all',min_target_points=1)
    assert len(result['train_probe'])==764 and len(result['dev'])==96
    assert not result['train_probe'].scenes & result['dev'].scenes
    return result


def parent(a,cfg):
    path=a.parent_root/'training_logs/full64-budget-001/checkpoints/epoch-0014.pt'
    assert sha(path)==PARENT_SHA
    state=load_warm_start(path,cfg,PARENT_IMPLEMENTATION)
    assert state['completed_epoch']==14 and state['global_step']==2674
    assert state['metadata']==read(a.parent_root/'training_logs/full64-budget-001/parent_metadata.json')
    return state


def score_arrays(gt,gc,raw,pc,final,fc,prob,roi,valid,full,pan,scene,index,assigned=None,ri=None,fi=None):
    ri=cpu_iou(gt,raw) if ri is None else ri
    fi=cpu_iou(gt,final) if fi is None else fi
    gp,cp,fp=pairs(ri,gc,pc,True),pairs(ri,gc,pc),pairs(fi,gc,fc)
    # The legacy Hungarian solver independently confirms all three cardinalities.
    for mat,cls,geometric,found in ((ri,pc,True,gp),(ri,pc,False,cp),(fi,fc,False,fp)):
        _,native=match_sets(mat,np.zeros(len(gc)) if geometric else gc,
                            np.zeros(len(cls)) if geometric else cls)
        assert len(native)==len(found)
    hits={g for g,q in fp};geometry_hits={g for g,q in gp};class_hits={g for g,q in cp}
    targets=[]
    for g,size in enumerate(gt.sum(1)):
        qualified=np.flatnonzero(ri[g]>=.5)
        correct=qualified[pc[qualified]==gc[g]]
        best=int(ri[g].argmax())
        row=dict(gt=g,points=int(size),class_id=int(gc[g]),bucket=bucket(size),matched=g in hits,
            raw_geometry_hit=g in geometry_hits,raw_class_hit=g in class_hits,
            geometry_exists=bool(len(qualified)),correct_class_exists=bool(len(correct)),
            all_qualified_no_object=bool(len(qualified) and np.all(pc[qualified]==16)),
            qualified_queries=qualified.tolist(),correct_class_qualified=correct.tolist(),
            best_query=best,best_iou=float(ri[g,best]),best_query_class=int(pc[best]),
            correct_minus_no_object=float(prob[best,gc[g]+1]-prob[best,-1]),
            scene_token=scene,frame=index)
        if assigned is not None:
            q=int(assigned[g]);row.update(matched_query=q,matched_query_class=int(pc[q]),
                matched_query_iou=float(ri[g,q]),matched_query_correct_probability=float(prob[q,gc[g]+1]),
                matched_query_no_object_probability=float(prob[q,-1]))
        targets.append(row)
    duplicates=sum(max(0,len(t['correct_class_qualified'])-1) for t in targets)
    row=dict(dataset_index=index,scene_token=scene,gt=len(gt),pred=len(final),tp=len(fp),fp=len(final)-len(fp),
        raw_geometry_tp=len(gp),raw_class_tp=len(cp),predicted_points=int(full.sum()),
        ignored_points=int(full[:,roi&~valid].sum()),ego_points=int(full[:,pan//1000==31].sum()),
        extra_correct_query_target_edges=duplicates,query_class_counts={str(k):int(v) for k,v in Counter(pc.tolist()).items()})
    return row,targets


def summary(rows,targets):
    keys=('gt','pred','tp','fp','raw_geometry_tp','raw_class_tp','predicted_points','ignored_points','ego_points','extra_correct_query_target_edges')
    result={k:sum(r[k] for r in rows) for k in keys}
    result.update(frames=len(rows),precision=result['tp']/max(1,result['pred']),recall=result['tp']/max(1,result['gt']),
                  ignored_fraction=result['ignored_points']/max(1,result['predicted_points']))
    groups={}
    for field in ('bucket','class_id','scene_token'):
        bins=defaultdict(list)
        for t in targets:bins[str(t[field])].append(t)
        groups[field]={k:dict(gt=len(v),tp=sum(t['matched'] for t in v),
            raw_geometry_tp=sum(t['raw_geometry_hit'] for t in v),raw_class_tp=sum(t['raw_class_hit'] for t in v),
            geometry_exists=sum(t['geometry_exists'] for t in v),all_qualified_no_object=sum(t['all_qualified_no_object'] for t in v),
            recall=sum(t['matched'] for t in v)/len(v)) for k,v in bins.items()}
    small=[groups['bucket'].get(k,dict(gt=0,tp=0,raw_geometry_tp=0,raw_class_tp=0)) for k in ('1-2','3-10')]
    groups['bucket']['1-10']={k:sum(v[k] for v in small) for k in ('gt','tp','raw_geometry_tp','raw_class_tp')}
    result['strata']=groups
    return result


def source_frame(a,ds,index):
    cpu=ds[index]['frames'][0]
    record=ds.records[index]['frames'][0]
    with np.load(a.data/'nuScenes-panoptic-v1.0-all'/record['source_panoptic_file']['path']) as z:pan=z['data'].reshape(-1)
    return cpu,record,pan


def save_frame(dest,arrays,row,targets,source):
    dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'prediction.npz',**arrays)
    row.update(token=source['sample_data_token'],source=source,snapshot_sha256=sha(dest/'prediction.npz'))
    write(dest/'targets.json',targets)
    return row


def finish_scope(folder,rows,targets,identity):
    write(folder/'metrics.json',summary(rows,targets))
    write(folder/'completion.json',dict(complete=True,frames=len(rows),identity=identity))
    return read(folder/'metrics.json')


def check_sources(a,plan):
    assert sha(__file__)==plan['script_sha256'] and implementation_digest()==plan['implementation_sha256']
    root=Path(__file__).resolve().parents[2]
    for name,expected in plan['source_files'].items():assert sha(root/name)==expected,name
    assert sha(a.parent_root/'training_logs/full64-budget-001/checkpoints/epoch-0014.pt')==PARENT_SHA


def prepare(a):
    assert not (a.out/'plan.json').exists()
    a.out.mkdir(parents=True,exist_ok=True)
    e1=read(a.e1/'completion.json');assert e1['full_e1_complete'] and e1['training_updates']==0
    old_plan=read(a.parent_root/'training_logs/full64-budget-001/plan.json')
    root=Path(__file__).resolve().parents[2]
    changed=[];files={}
    for name,expected in old_plan['native_files'].items():
        assert sha(a.parent_root/name)==expected
        files[name]=sha(root/name)
        if files[name]!=expected:changed.append(name)
    assert set(changed)=={'mllm/maskpls_b4dl/'+n for n in ('config.py','criterion.py','runner.py')}
    cfg=MaskPLSConfig.from_dict(old_plan['config']);state=parent(a,cfg)
    assert cfg.mask_ignore_policy=='background' and cfg.mask_bce_reduction=='point_mean'
    assert cfg.num_queries==128 and cfg.no_object_weight==.1
    assert (cfg.score_threshold,cfg.mask_threshold,cfg.overlap_threshold)==(.25,.5,.8)
    tv=a.parent_root/'datageneration/data/trainval-64-001'
    for filename,key in [('train.jsonl','train_manifest_sha256'),('internal_dev.jsonl','dev_manifest_sha256')]:
        assert sha(tv/filename)==old_plan[key]
    for data in (state['metadata']['data'],state['metadata']['internal_dev_data']):
        for name,expected in data['source_split']['metadata_sha256'].items():
            base=a.data/'nuScenes-panoptic-v1.0-all' if name=='panoptic' else a.data
            assert sha(base/'v1.0-trainval'/(name+'.json'))==expected
    del state
    ds=datasets(a,cfg);order=np.stack([np.random.default_rng(2026+e).permutation(764) for e in range(3)])
    np.save(a.out/'frame_order.npy',order)
    plan=dict(experiment='E2-C',source_commit=a.source_commit,script_sha256=sha(__file__),source_files=files,
        implementation_sha256=implementation_digest(),parent_checkpoint_sha256=PARENT_SHA,
        parent_implementation_sha256=PARENT_IMPLEMENTATION,config=cfg.to_dict(),arms=ARMS,
        epochs_per_arm=3,updates_per_epoch=191,updates_per_arm=573,total_planned_updates=1146,
        frame_exposures_per_arm=2292,total_planned_frame_exposures=4584,
        recipe=dict(seed=2026,lr=1e-4,weight_decay=.01,accumulate=4,gradient_clip=1.,precision='FP32',
            batch_size=1,scheduler='fresh StepLR(80,0.1), constant over three epochs',trainable='all parameters'),
        sampling='pre-generated CPU indices, identical NPZ for both arms; no model RNG dependence',
        frame_order_sha256=sha(a.out/'frame_order.npy'),train_manifest_sha256=old_plan['train_manifest_sha256'],
        dev_manifest_sha256=old_plan['dev_manifest_sha256'],train_probe_indices=old_plan['train_probe_indices'],
        dev_indices=list(range(96)),ignored_raw_semantic_ids=old_plan['ignored_raw_semantic_ids'],
        zero_epoch='reuse first audited E1 epoch14 masks for both arms; objective weight does not enter forward',
        e1_completion_sha256=sha(a.e1/'completion.json'),e1_identity_sha256=sha(a.e1/'identity.json'),
        metric='IoU>=0.5 max cardinality then max total IoU; SciPy linear_sum_assignment and stable input order for ALL strata',
        primary_endpoint=3,eval_new_forwards=960,extra_smoke_forwards=2,
        stop='per arm: two consecutive epoch1/2/3 evaluations below half F TP or small geometry; epoch0 excluded',
        scope='Adaptive exposed internal dev; single seed follow-up from common epoch14, not independent retraining or new holdout',
        gates=dict(final_tp_margin=10,precision_absolute_drop=.02,small_final_tp_drop=0,small_geometry_tp_drop=10,
            medium_large_recall_drop=.02,ignored_fraction_increase=.01,ego_multiplier=1.1,ego_add=100,min_scenes_nondecreasing=5))
    treatment=dict(plan['config'],no_object_weight=.05)
    assert [k for k in treatment if treatment[k]!=plan['config'][k]]==['no_object_weight']
    # Cache all point samples before training; files also exercise every train frame's hash/data contract.
    manifest=[]
    for epoch in range(3):
        for ordinal,index in enumerate(order[epoch].tolist()):
            frame=ds['train_probe'][index]['frames'][0]
            xyz=frame['points'][:,:3];bounds=xyz.new_tensor(cfg.point_cloud_range)
            valid=((xyz>=bounds[:3])&(xyz<bounds[3:])).all(1)
            labels,classes,_=build_targets(frame,valid);labels=labels.clamp_min(0)
            assert len(labels)<=128
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(202600000+epoch*764+ordinal)
                mi=sample_supervised_points(labels,valid,cfg.match_points,cfg.points_per_instance)
                li=sample_supervised_points(labels,valid,cfg.loss_points,cfg.points_per_instance)
            path=a.out/'sampling'/f'epoch-{epoch+1:02d}'/f'frame-{index:04d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,match=array(mi),loss=array(li))
            row=dict(epoch=epoch+1,ordinal=ordinal,index=index,relative_path=str(path.relative_to(a.out)),sha256=sha(path))
            manifest.append(row)
            if ordinal%100==0:write(a.out/'preparation_progress.json',dict(stage='sampling',epoch=epoch+1,completed=ordinal+1,expected=764))
    write(a.out/'sampling_manifest.json',manifest);plan['sampling_manifest_sha256']=sha(a.out/'sampling_manifest.json')
    cohort=[]
    # Re-score saved F masks under the exact E2 matcher, avoiding cross-solver bucket drift.
    for scope,indices in [('train_probe',plan['train_probe_indices']),('dev',plan['dev_indices'])]:
        rows=[];targets=[];folder=a.out/'F'/scope
        for index in indices:
            frame,src,pan=source_frame(a,ds[scope],index)
            origin=a.e1/f'epoch-0014-{scope}'/f'frame-{index:04d}'
            old_row=next(r for r in map(json.loads,(origin.parent/'frames.jsonl').read_text().splitlines()) if r['dataset_index']==index)
            assert sha(origin/'snapshot.npz')==old_row['snapshot_sha256']
            with np.load(origin/'snapshot.npz') as z:
                valid=z['valid'];roi=z['roi'];gc=z['gt_class'];pc=z['query_class'];prob=z['layer_class_probability'][-1]
                raw=np.unpackbits(z['layer_raw_packed'][-1],axis=1,count=int(valid.sum()),bitorder='little').astype(bool)
                full=np.unpackbits(z['final_full_packed'],axis=1,count=len(valid),bitorder='little').astype(bool)
                fc=z['final_class'];assigned=z['matching_assignment'][-1,:len(gc)] if scope=='train_probe' else None
            gt=array(frame['labels'].eq(1))[:,valid]
            row,selected=score_arrays(gt,gc,raw,pc,full[:,valid],fc,prob,roi,valid,full,pan,frame['scene_token'],index,assigned)
            arrays=dict(valid=valid,roi=roi,gt_class=gc,query_class=pc,class_probability=prob,
                raw_packed=np.packbits(raw,axis=1,bitorder='little'),final_full_packed=np.packbits(full,axis=1,bitorder='little'),final_class=fc)
            if assigned is not None:arrays['matching_assignment']=assigned
            row=save_frame(folder/f'frame-{index:04d}',arrays,row,selected,src)
            log(folder/'frames.jsonl',row);rows.append(row);targets.extend(selected)
            if scope=='train_probe':
                cohort.extend(dict(frame=index,gt=t['gt']) for t in selected if t['all_qualified_no_object'] and t['matched_query_iou']>=.5)
        finish_scope(folder,rows,targets,dict(kind='E1_cached_first_pass',checkpoint_sha256=PARENT_SHA,new_forwards=0))
    assert len(cohort)==167
    write(a.out/'matched_no_object_cohort.json',cohort)
    plan['baseline_dev']=read(a.out/'F/dev/metrics.json');plan['baseline_train_probe']=read(a.out/'F/train_probe/metrics.json')
    assert plan['baseline_dev']['tp']==188 and plan['baseline_dev']['gt']==3217
    plan['cpu_preflight_complete']=True
    write(a.out/'plan.json',plan);write(a.out/'preparation_completion.json',dict(complete=True,training_updates=0,sampling_files=len(manifest),baseline_cached_frames=160))
    print('E2C_PREPARATION_COMPLETE',flush=True)


def new_model(a,cfg):
    seed(2026);state=parent(a,cfg);model=MaskPLS(cfg).cuda()
    model.load_state_dict(state['model'],strict=True);del state
    return model


def samples(a,item,device):
    path=a.out/item['relative_path'];assert sha(path)==item['sha256']
    with np.load(path) as z:return tuple(torch.as_tensor(z[k].copy(),dtype=torch.long,device=device) for k in ('match','loss'))


def smoke(a):
    plan=read(a.out/'plan.json');check_sources(a,plan)
    item=read(a.out/'sampling_manifest.json')[0];records=[];initial=[]
    for arm,weight in ARMS.items():
        cfg=MaskPLSConfig.from_dict(dict(plan['config'],no_object_weight=weight));model=new_model(a,cfg)
        initial.append(digest(model));ds=datasets(a,cfg);cpu=ds['train_probe'][item['index']]['frames'][0]
        frame=to_device(cpu,torch.device('cuda:0'));mi,li=samples(a,item,frame['points'].device)
        model.train();torch.cuda.reset_peak_memory_stats();output=model(frame['points'])
        loss,parts,assignment=MaskPLSCriterion(cfg)(output,frame,match_indices=mi,loss_indices=li)
        # Compare the default objective against the parent's actual criterion on identical tensors/samples.
        if arm=='C0':
            path=a.parent_root/'mllm/maskpls_b4dl/criterion.py'
            spec=importlib.util.spec_from_file_location('maskpls_b4dl._e2c_parent_criterion',path)
            legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
            with patch.object(legacy,'sample_supervised_points',side_effect=[mi,li]):
                old_loss,old_parts,old_assignment=legacy.MaskPLSCriterion(cfg)(output,frame)
            assert torch.equal(loss,old_loss) and assignment==old_assignment
            assert all(torch.equal(parts[k],old_parts[k]) for k in parts)
            other=MaskPLSCriterion(MaskPLSConfig.from_dict(dict(plan['config'],no_object_weight=.05)))
            _,other_parts,other_assignment=other(output,frame,match_indices=mi,loss_indices=li)
            assert other_assignment==assignment
            assert all(torch.equal(parts[k],other_parts[k]) for k in parts if k!='classification')
            del old_loss,old_parts,old_assignment,other_parts,other,legacy
        assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        assert any(p.grad is not None for p in model.parameters())
        records.append(dict(arm=arm,loss=float(loss.detach()),gradient_norm=float(norm),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),optimizer_updates=0))
        del model,output,loss,parts;torch.cuda.empty_cache()
    assert initial[0]==initial[1]
    write(a.out/'smoke.json',dict(complete=True,extra_forwards=2,training_updates=0,initial_state_sha256=initial[0],
        default_loss_exactly_matches_parent=True,only_classification_changed=True,records=records))


@torch.no_grad()
def evaluate(a,model,ds,scope,indices,folder,identity):
    cfg=model.config;model.eval();model.requires_grad_(False);before=digest(model)
    rows=[];all_targets=[]
    for ordinal,index in enumerate(indices):
        cpu,src,pan=source_frame(a,ds,index);frame=to_device(cpu,torch.device('cuda:0'));output=model(frame['points'])
        native=extract_instances(output,cfg);valid_t=output['valid_points']&frame['semantic_labels'].ne(0)
        raw_t=output['mask_logits'].sigmoid().ge(cfg.mask_threshold)[:,valid_t]
        final_t=native['masks'][:,valid_t];gt_t=frame['labels'].eq(1)[:,valid_t]
        valid=array(valid_t);roi=array(output['valid_points']);full=array(native['masks']);gc=array(frame['class_ids'])
        raw=array(raw_t);fc=array(native['class_ids']);prob=array(output['class_logits'].softmax(-1));pc=prob.argmax(1)-1
        assignment=None
        if scope=='train_probe':
            labels,classes,_=build_targets(frame,output['valid_points']);labels=labels.clamp_min(0)
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(2026+index)
                mi=sample_supervised_points(labels,output['valid_points'],cfg.match_points,cfg.points_per_instance)
            assignment=np.asarray(MaskPLSCriterion(cfg).match(output,labels,classes,mi))[:len(gc)]
        row,targets=score_arrays(array(gt_t),gc,raw,pc,array(final_t),fc,prob,roi,valid,full,pan,cpu['scene_token'],index,
            assignment,gpu_iou(gt_t,raw_t),gpu_iou(gt_t,final_t))
        arrays=dict(valid=valid,roi=roi,gt_class=gc,query_class=pc,class_probability=prob,
            raw_packed=np.packbits(raw,axis=1,bitorder='little'),final_full_packed=np.packbits(full,axis=1,bitorder='little'),final_class=fc)
        if assignment is not None:arrays['matching_assignment']=assignment
        row=save_frame(folder/f'frame-{index:04d}',arrays,row,targets,src)
        log(folder/'frames.jsonl',row);rows.append(row);all_targets.extend(targets)
        write(a.out/'progress.json',dict(stage='evaluation',identity=identity,scope=scope,completed=ordinal+1,expected=len(indices)))
        del output,frame,native
    assert before==digest(model)
    model.requires_grad_(True)
    return finish_scope(folder,rows,all_targets,dict(**identity,state_sha256=before,eval_state_unchanged=True))


def train_arm(a,arm):
    plan=read(a.out/'plan.json');check_sources(a,plan);assert read(a.out/'smoke.json')['complete']
    folder=a.out/arm;folder.mkdir(exist_ok=False)
    cfg=MaskPLSConfig.from_dict(dict(plan['config'],no_object_weight=ARMS[arm]));ds=datasets(a,cfg);model=new_model(a,cfg)
    assert digest(model)==read(a.out/'smoke.json')['initial_state_sha256']
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=80,gamma=.1)
    assert len(optimizer.state)==0 and all(p.requires_grad for p in model.parameters())
    metadata=base_metadata(cfg);metadata.update(e2c_plan_sha256=sha(a.out/'plan.json'),arm=arm,
        initialization=dict(kind='audited_epoch14_model_only',checkpoint_sha256=PARENT_SHA,
            source_implementation_sha256=PARENT_IMPLEMENTATION,optimizer_reset=True,scheduler_reset=True),
        recipe=plan['recipe'],train_manifest_sha256=plan['train_manifest_sha256'],dev_manifest_sha256=plan['dev_manifest_sha256'])
    origin=read(a.parent_root/'training_logs/full64-budget-001/parent_metadata.json')
    metadata.update(data=origin['data'],internal_dev_data=origin['internal_dev_data'],
        pretrained_official_train_exposure=origin.get('pretrained_official_train_exposure',False),
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                     minkowski_engine='0.5.4',knn_backend=cfg.knn_backend))
    metadata['initialization'].update(source_config=origin['config'],source_completed_epoch=14,
        source_global_step=2674,source_initialization=origin.get('initialization'))
    write(folder/'initialization.json',dict(initial_state_sha256=digest(model),optimizer_states=0,
        scheduler_last_epoch=scheduler.last_epoch,trainable_parameters=sum(p.numel() for p in model.parameters()),metadata=metadata))
    manifest=read(a.out/'sampling_manifest.json');step=0;streak=0
    for epoch in range(1,4):
        model.train();items=[r for r in manifest if r['epoch']==epoch];assert len(items)==764
        for offset in range(0,764,4):
            window=items[offset:offset+4];optimizer.zero_grad(set_to_none=True);parts=Counter();start=time.monotonic()
            for item in window:
                seed(402600000+(epoch-1)*764+item['ordinal'])
                cpu=ds['train_probe'][item['index']]['frames'][0];frame=to_device(cpu,torch.device('cuda:0'))
                mi,li=samples(a,item,frame['points'].device)
                output=model(frame['points']);loss,components,assignment=MaskPLSCriterion(cfg)(output,frame,match_indices=mi,loss_indices=li)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                (loss/4).backward()
                for k,v in components.items():parts[k]+=float(v.detach())/4
                log(folder/'training_frames.jsonl',dict(epoch=epoch,ordinal=item['ordinal'],dataset_index=item['index'],
                    sampling_sha256=item['sha256'],loss=float(loss.detach())))
                del frame,output,loss,components,mi,li
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            record=dict(arm=arm,epoch=epoch,global_step=step,gradient_norm=float(norm),seconds=time.monotonic()-start,**parts)
            log(folder/'steps.jsonl',record)
            write(a.out/'progress.json',dict(stage='training',arm=arm,epoch=epoch,global_step=step,expected_steps=573,
                completed_frame_exposures=(epoch-1)*764+offset+4,last_step_seconds=record['seconds']))
            if step%10==0:print(json.dumps(record),flush=True)
        optimizer.zero_grad(set_to_none=True);scheduler.step()
        assert step==epoch*191 and all(g['lr']==1e-4 for g in optimizer.param_groups)
        checkpoint=folder/f'epoch-{epoch:02d}.pt';temp=checkpoint.with_suffix('.pt.part')
        torch.save(dict(metadata=metadata,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            rng=rng_state(),completed_epoch=epoch,global_step=step),temp);temp.replace(checkpoint)
        identity=dict(arm=arm,epoch=epoch,global_step=step,checkpoint_sha256=sha(checkpoint))
        for scope,indices in [('train_probe',plan['train_probe_indices']),('dev',plan['dev_indices'])]:
            evaluate(a,model,ds[scope],scope,indices,folder/f'eval-{epoch:02d}'/scope,identity)
        dev=read(folder/f'eval-{epoch:02d}/dev/metrics.json');baseline=plan['baseline_dev']
        bad=dev['tp']<baseline['tp']*.5 or dev['strata']['bucket']['1-10']['raw_geometry_tp']<baseline['strata']['bucket']['1-10']['raw_geometry_tp']*.5
        streak=streak+1 if bad else 0
        log(folder/'epochs.jsonl',dict(**identity,dev=dev,catastrophic_streak=streak,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        if streak>=2:
            write(folder/'completion.json',dict(status='stopped_early',reason='two consecutive catastrophic evaluations',
                complete=False,completed_epoch=epoch,updates=step,frame_exposures=epoch*764));break
    else:write(folder/'completion.json',dict(status='completed',complete=True,completed_epoch=3,updates=573,frame_exposures=2292))
    del model,optimizer,scheduler;torch.cuda.empty_cache();check_sources(a,plan)


def audit(a):
    plan=read(a.out/'plan.json');check_sources(a,plan);cfg=MaskPLSConfig.from_dict(plan['config'])
    scopes=[]
    for folder in sorted(a.out.glob('*/eval-*/*'))+list((a.out/'F').glob('*')):
        if not (folder/'frames.jsonl').exists():continue
        scope=folder.name;expected=plan['train_probe_indices'] if scope=='train_probe' else plan['dev_indices']
        records=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
        assert [r['dataset_index'] for r in records]==expected
        checked=[];targets=[]
        for row in records:
            src=row['source'];idx=row['dataset_index'];dest=folder/f'frame-{idx:04d}'
            tv=a.parent_root/'datageneration/data/trainval-64-001';panroot=a.data/'nuScenes-panoptic-v1.0-all'
            for base,key in ((a.data,'point_file'),(tv,'mask_file'),(panroot,'source_panoptic_file')):
                assert sha(base/src[key]['path'])==src[key]['sha256']
            xyz=np.fromfile(a.data/src['point_file']['path'],dtype='<f4').reshape(-1,5)[:,:3]
            with np.load(tv/src['mask_file']['path']) as z:labels=z['labels']==1
            with np.load(panroot/src['source_panoptic_file']['path']) as z:pan=z['data'].reshape(-1)
            bounds=np.asarray(cfg.point_cloud_range);roi=((xyz>=bounds[:3])&(xyz<bounds[3:])).all(1)
            valid=roi&~np.isin(pan//1000,plan['ignored_raw_semantic_ids'])
            assert sha(dest/'prediction.npz')==row['snapshot_sha256']
            with np.load(dest/'prediction.npz') as z:
                assert np.array_equal(roi,z['roi']) and np.array_equal(valid,z['valid'])
                assert np.array_equal(z['gt_class'],src['class_ids'])
                raw=np.unpackbits(z['raw_packed'],axis=1,count=int(valid.sum()),bitorder='little').astype(bool)
                full=np.unpackbits(z['final_full_packed'],axis=1,count=len(valid),bitorder='little').astype(bool)
                assert not full[:,~roi].any()
                assigned=z['matching_assignment'] if 'matching_assignment' in z else None
                new,selected=score_arrays(labels[:,valid],z['gt_class'],raw,z['query_class'],full[:,valid],z['final_class'],
                    z['class_probability'],roi,valid,full,pan,row['scene_token'],idx,assigned)
            assert all(row[k]==v for k,v in new.items())
            stored=read(dest/'targets.json');assert stored==selected
            checked.append(new);targets.extend(selected)
        assert summary(checked,targets)==read(folder/'metrics.json')
        scopes.append(dict(path=str(folder.relative_to(a.out)),frames=len(checked),complete=True))
        print('AUDITED',str(folder),flush=True)
    expected_scopes=2+sum(read(a.out/arm/'completion.json')['completed_epoch']*2 for arm in ARMS)
    assert len(scopes)==expected_scopes
    # Verify every actually consumed sampling file/order, not just RNG seeds.
    sampling=read(a.out/'sampling_manifest.json')
    for arm in ARMS:
        done=read(a.out/arm/'completion.json');rows=[json.loads(s) for s in (a.out/arm/'training_frames.jsonl').read_text().splitlines()]
        expected=sampling[:done['frame_exposures']]
        assert [(r['epoch'],r['ordinal'],r['dataset_index'],r['sampling_sha256']) for r in rows]==[
            (r['epoch'],r['ordinal'],r['index'],r['sha256']) for r in expected]
        steps=[json.loads(s) for s in (a.out/arm/'steps.jsonl').read_text().splitlines()]
        assert [r['global_step'] for r in steps]==list(range(1,done['updates']+1))
        for e in range(1,done['completed_epoch']+1):
            ckpt=load_native(a.out/arm/f'epoch-{e:02d}.pt')
            assert ckpt['global_step']==191*e and ckpt['metadata']['e2c_plan_sha256']==sha(a.out/'plan.json')
            counters=[int(s['step']) for s in ckpt['optimizer']['state'].values()]
            assert counters and min(counters)==max(counters)==191*e
            del ckpt
    write(a.out/'audit.json',dict(complete=True,scopes=scopes,total_evaluation_frames=sum(r['frames'] for r in scopes),
        independent_cpu_iou_and_matching_cardinality=True,all_source_point_axis_and_prediction_hashes=True,
        paired_sampling_order_and_update_counts_verified=True,parent_unchanged=True))


def finalize(a):
    plan=read(a.out/'plan.json');assert read(a.out/'audit.json')['complete'];check_sources(a,plan)
    done={arm:read(a.out/arm/'completion.json') for arm in ARMS};gates={}
    if all(d['complete'] for d in done.values()):
        f=plan['baseline_dev'];c=read(a.out/'C0/eval-03/dev/metrics.json');t=read(a.out/'C1/eval-03/dev/metrics.json')
        fb,cb,tb=[m['strata']['bucket'] for m in (f,c,t)]
        scenes=t['strata']['scene_token'];control=c['strata']['scene_token']
        gates=dict(final_tp=t['tp']>=max(c['tp'],f['tp'])+10,precision=t['precision']>=max(c['precision'],f['precision'])-.02,
            small_final=tb['1-10']['tp']>=max(cb['1-10']['tp'],fb['1-10']['tp']),
            small_geometry=tb['1-10']['raw_geometry_tp']>=max(cb['1-10']['raw_geometry_tp'],fb['1-10']['raw_geometry_tp'])-10,
            medium_recall=tb['11-49']['recall']>=cb['11-49']['recall']-.02,
            large_recall=tb['50+']['recall']>=cb['50+']['recall']-.02,
            ignored=t['ignored_fraction']<=max(c['ignored_fraction'],f['ignored_fraction'])+.01,
            ego=t['ego_points']<=max(c['ego_points'],f['ego_points'])*1.1+100,
            scenes=sum(v['recall']>=control[k]['recall'] for k,v in scenes.items())>=5)
        write(a.out/'comparison.json',dict(F=f,C0=c,C1=t,gates=gates,screen_passed=all(gates.values()),
            caveat='Engineering screen on exposed development scenes, not statistical significance or heldout acceptance'))
    cohort=read(a.out/'matched_no_object_cohort.json');converted={}
    for arm in ARMS:
        epoch=done[arm]['completed_epoch'];rows=[]
        for case in cohort:
            t=read(a.out/arm/f'eval-{epoch:02d}/train_probe'/f"frame-{case['frame']:04d}"/'targets.json')[case['gt']]
            rows.append(dict(**case,geometry_exists=t['geometry_exists'],correct_class_exists=t['correct_class_exists'],matched=t['matched']))
        converted[arm]=dict(n=len(rows),geometry_exists=sum(r['geometry_exists'] for r in rows),
            correct_class_exists=sum(r['correct_class_exists'] for r in rows),final_tp=sum(r['matched'] for r in rows),records=rows)
    write(a.out/'cohort_conversion.json',converted)
    write(a.out/'completion.json',dict(complete=True,planned_endpoint_completed=all(d['complete'] for d in done.values()),
        actual_optimizer_updates=sum(d['updates'] for d in done.values()),actual_training_frame_exposures=sum(d['frame_exposures'] for d in done.values()),
        extra_smoke_forwards=2,arms=done,audit_complete=True,screen_passed=bool(gates) and all(gates.values()),
        status='numeric_complete_report_pending',human_reviewed=False,new_holdout_evaluated=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','smoke','run','audit','finalize'])
    p.add_argument('--parent-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--e1',type=Path,required=True);p.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/nuScenes'))
    p.add_argument('--source-commit',default='recorded in deployment identity')
    a=p.parse_args()
    try:
        if a.mode=='prepare':prepare(a)
        elif a.mode=='smoke':smoke(a)
        elif a.mode=='run':
            for arm in ARMS:train_arm(a,arm)
        elif a.mode=='audit':audit(a)
        else:finalize(a)
    except Exception as e:
        write(a.out/'failure.json',dict(mode=a.mode,error_type=type(e).__name__,error=str(e)))
        raise


if __name__=='__main__':main()
