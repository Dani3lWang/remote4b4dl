"""Preregistered E2-Q: CPU prepare, real CUDA smoke, paired training and audit.

Run with the verified wqlc environment in an independent deployment directory.
The E2-C helper is a frozen source dependency for data/metric/NPZ contracts.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

from maskpls_b4dl.config import MaskPLSConfig
from maskpls_b4dl.criterion import MaskPLSCriterion, sample_supervised_points
from maskpls_b4dl.data import build_targets
from maskpls_b4dl.e2q import (ARMS, QUERY_KEYS, configure_trainability,
                             initialize_model, preserve_model_modes, selection_gates,
                             tensor_digest)
from maskpls_b4dl.model import MaskPLS
from maskpls_b4dl.protocol import implementation_digest
from maskpls_b4dl.runner import base_metadata, load_native
from grounded4d.train import to_device, rng_state

_spec = importlib.util.spec_from_file_location("e2q_metric_dependency", Path(__file__).with_name("run_maskpls_e2c.py"))
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)


def source_files():
    root = Path(__file__).resolve().parents[2]
    files = list((root/"mllm/maskpls_b4dl").glob("*.py"))
    files += list((root/"mllm/grounded4d").glob("*.py"))
    files += [Path(__file__).resolve(),Path(__file__).with_name("run_maskpls_e2c.py"),
              Path(__file__).with_name("run_maskpls_e2q_workflow.sh")]
    return {p.relative_to(root).as_posix(): h.sha(p) for p in sorted(files)}


def check_sources(a, plan):
    if source_files() != plan["source_files"] or implementation_digest() != plan["implementation_sha256"]:
        raise ValueError("E2-Q deployed source identity changed")
    path = a.parent_root/"training_logs/full64-budget-001/checkpoints/epoch-0014.pt"
    if h.sha(path) != h.PARENT_SHA:
        raise ValueError("parent checkpoint identity changed")
    tv = a.parent_root/"datageneration/data/trainval-64-001"
    for name,key in (("train.jsonl","train_manifest_sha256"),("internal_dev.jsonl","dev_manifest_sha256")):
        if h.sha(tv/name) != plan[key]:
            raise ValueError("dataset manifest identity changed")
    for name,key in (("frame_order.npy","frame_order_sha256"),("sampling_manifest.json","sampling_manifest_sha256")):
        if h.sha(a.out/name) != plan[key]:
            raise ValueError("paired data stream identity changed")


def parent_config(a):
    old = h.read(a.parent_root/"training_logs/full64-budget-001/plan.json")
    cfg = MaskPLSConfig.from_dict(old["config"])
    if (cfg.num_queries,cfg.voxel_size,cfg.mask_ignore_policy,cfg.mask_bce_reduction,
        cfg.no_object_weight,cfg.query_initialization,cfg.score_threshold,cfg.mask_threshold,
        cfg.overlap_threshold) != (128,.05,"background","point_mean",.1,"learned",.25,.5,.8):
        raise ValueError("unexpected parent recipe")
    return old,cfg


def new_model(a, arm=None):
    _,base = parent_config(a)
    # Validate the native parent under its ORIGINAL learned-mode configuration.
    # Query migration below is an explicit experiment, not a resume bypass.
    state = h.parent(a,base)
    cfg = MaskPLSConfig.from_dict(dict(base.to_dict(),query_initialization=ARMS[arm] if arm else "learned"))
    h.seed(2026)
    model = MaskPLS(cfg)
    if arm:
        identity = initialize_model(model,state["model"],arm)
    else:
        model.load_state_dict(state["model"],strict=True)
        identity = dict(kind="frozen_epoch14",initial_state_sha256=tensor_digest(model.state_dict()))
    del state
    return model.cuda(),identity


def prepare(a):
    if a.out.exists():
        raise ValueError("prepare requires a NEW output directory")
    old,cfg = parent_config(a)
    state = h.parent(a,cfg)
    metadata = state["metadata"]
    # Use audited F metrics as a reference only; re-run F on deployment hardware.
    reference_done = h.read(a.reference/"completion.json")
    if not reference_done.get("audit_complete") or not h.read(a.reference/"audit.json")["complete"]:
        raise ValueError("reference E2-C audit is incomplete")
    reference_f = h.read(a.reference/"F/dev/metrics.json")
    if (reference_f["frames"],reference_f["gt"],reference_f["tp"]) != (96,3217,188):
        raise ValueError("reference F does not match the frozen baseline")
    tv = a.parent_root/"datageneration/data/trainval-64-001"
    for filename,key in (("train.jsonl","train_manifest_sha256"),("internal_dev.jsonl","dev_manifest_sha256")):
        if h.sha(tv/filename) != old[key]:
            raise ValueError("parent manifest identity mismatch")
    for data in (metadata["data"],metadata["internal_dev_data"]):
        for name,expected in data["source_split"]["metadata_sha256"].items():
            base = a.data/"nuScenes-panoptic-v1.0-all" if name == "panoptic" else a.data
            if h.sha(base/"v1.0-trainval"/(name+".json")) != expected:
                raise ValueError("source nuScenes metadata changed")
    parent_sha = tensor_digest(state["model"])
    del state
    a.out.mkdir(parents=True)
    ds = h.datasets(a,cfg)
    order = np.stack([np.random.default_rng(2026+e).permutation(764) for e in range(6)])
    np.save(a.out/"frame_order.npy",order)
    plan = dict(experiment="E2-Q",schema="b4dl-maskpls-e2q-v1",source_commit=a.source_commit,
        source_files=source_files(),implementation_sha256=implementation_digest(),
        parent_checkpoint_sha256=h.PARENT_SHA,parent_implementation_sha256=h.PARENT_IMPLEMENTATION,
        parent_state_sha256=parent_sha,config=cfg.to_dict(),arms=ARMS,
        parent_metadata=metadata,reset_parameters=list(QUERY_KEYS),
        train_manifest_sha256=old["train_manifest_sha256"],dev_manifest_sha256=old["dev_manifest_sha256"],
        train_probe_indices=old["train_probe_indices"],dev_indices=list(range(96)),
        ignored_raw_semantic_ids=old["ignored_raw_semantic_ids"],frame_order_sha256=h.sha(a.out/"frame_order.npy"),
        epochs_per_arm=6,updates_per_epoch=191,updates_per_arm=1146,total_planned_updates=3438,
        frame_exposures_per_arm=4584,total_planned_frame_exposures=13752,
        evaluation_epochs=[0,2,4,6],primary_endpoint=6,
        planned_evaluation_forwards=2080,planned_smoke_forwards=3,
        reference_completion_sha256=h.sha(a.reference/"completion.json"),reference_F_dev=reference_f,
        recipe=dict(seed=2026,lr=1e-4,weight_decay=.01,batch_size=1,accumulate=4,
                    gradient_clip=1.,precision="FP32",scheduler="constant 1e-4 for six epochs",
                    trainable="shared full network; Q0 both tables, Q1 position only, Q2 neither table"),
        fps=dict(input="ROI-only raw LiDAR XYZ in metres; no GT/camera/answer",
                 start="farthest from sensor origin",duplicates="first original XYZ row",
                 ties="first original row",shortfall="cycle completed unique-point FPS sequence",
                 distance="CPU FP64",encoding="existing PositionalEncoder x/48,y/48,z/4; no added projection"),
        metric="IoU>=0.5 maximum cardinality then maximum total IoU; independent native cardinality check",
        stop="nonfinite/identity/point-axis/GT-leak failure immediately; two consecutive eval2/4/6 below half F TP or small geometric TP; eval0 excluded",
        gates=dict(final_tp_margin=10,precision_absolute_drop=.02,small_final_tp_drop=0,
                   small_geometry_tp_margin=20,medium_large_recall_drop=.02,
                   ignored_fraction_increase=.01,ego_multiplier=1.1,ego_add=100,min_scenes_nondecreasing=5),
        scope="Adaptive exposed internal dev; single paired follow-up seed from epoch14; no new holdout or language/temporal acceptance")
    h.write(a.out/"preregistration.json",plan)
    manifest = []
    for epoch,indices in enumerate(order,1):
        for ordinal,index in enumerate(indices.tolist()):
            frame = ds["train_probe"][index]["frames"][0]
            xyz = frame["points"][:,:3];bounds = xyz.new_tensor(cfg.point_cloud_range)
            valid = ((xyz>=bounds[:3])&(xyz<bounds[3:])).all(1)
            labels,_,_ = build_targets(frame,valid);labels = labels.clamp_min(0)
            if len(labels)>128:
                raise ValueError("query capacity exceeded")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(202600000+(epoch-1)*764+ordinal)
                samples = [sample_supervised_points(labels,valid,budget,cfg.points_per_instance)
                           for budget in (cfg.match_points,cfg.loss_points)]
            relative = Path("sampling")/f"epoch-{epoch:02d}"/f"frame-{index:04d}.npz"
            path = a.out/relative;path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,match=h.array(samples[0]),loss=h.array(samples[1]))
            manifest.append(dict(epoch=epoch,ordinal=ordinal,index=index,relative_path=relative.as_posix(),sha256=h.sha(path)))
            if ordinal%100==0:
                h.write(a.out/"progress.json",dict(stage="CPU preparation",epoch=epoch,completed=ordinal+1,expected=764))
                print("PREPARE",epoch,ordinal+1,flush=True)
    # Exercise the remaining 96 dev records' original point/label/panoptic hashes.
    for index in range(96):
        ds["dev"][index]
    h.write(a.out/"sampling_manifest.json",manifest)
    plan["sampling_manifest_sha256"] = h.sha(a.out/"sampling_manifest.json")
    h.write(a.out/"plan.json",plan)
    h.write(a.out/"preparation_completion.json",dict(complete=True,training_updates=0,
        train_frames=764,dev_frames=96,sampling_files=len(manifest),plan_sha256=h.sha(a.out/"plan.json")))


def enrich_targets(targets,gt,raw,xyz):
    """GT-derived distance/coverage/purity are offline diagnostics ONLY."""
    for target,mask in zip(targets,gt):
        distance = float(np.linalg.norm(xyz[mask].mean(0)[:2]))
        q = target["best_query"];inter = int((mask&raw[q]).sum())
        target.update(distance_m=distance,distance_bin=("0-10" if distance<10 else
            "10-20" if distance<20 else "20-40" if distance<40 else "40+"),
            best_query_coverage=inter/max(1,int(mask.sum())),
            best_query_purity=inter/max(1,int(raw[q].sum())))
    return targets


def summary(rows,targets):
    result = h.summary(rows,targets)
    distances = {}
    for name in ("0-10","10-20","20-40","40+"):
        group = [t for t in targets if t["distance_bin"]==name]
        distances[name] = dict(gt=len(group),tp=sum(t["matched"] for t in group),
            raw_geometry_tp=sum(t["raw_geometry_hit"] for t in group),
            raw_class_tp=sum(t["raw_class_hit"] for t in group),
            recall=sum(t["matched"] for t in group)/max(1,len(group)))
    result["strata"]["distance"] = distances
    result["geometry_diagnostics"] = {name:dict(
        gt=len(group),median_best_query_coverage=float(np.median([t["best_query_coverage"] for t in group])),
        median_best_query_purity=float(np.median([t["best_query_purity"] for t in group])))
        for name,group in [("all",targets)]+[(name,[t for t in targets if t["bucket"]==name])
                                            for name in ("1-2","3-10","11-49","50+")] if group}
    return result


@torch.no_grad()
def evaluate(a,model,ds,scope,indices,folder,identity):
    rows = [];all_targets = []
    with preserve_model_modes(model):
        model.eval();before = tensor_digest(model.state_dict())
        for ordinal,index in enumerate(indices):
            cpu,src,pan = h.source_frame(a,ds,index)
            frame = to_device(cpu,torch.device("cuda:0"));output = model(frame["points"])
            native = h.extract_instances(output,model.config)
            valid_t = output["valid_points"]&frame["semantic_labels"].ne(0)
            raw_t = output["mask_logits"].sigmoid().ge(model.config.mask_threshold)[:,valid_t]
            final_t = native["masks"][:,valid_t];gt_t = frame["labels"].eq(1)[:,valid_t]
            valid = h.array(valid_t);roi = h.array(output["valid_points"]);full = h.array(native["masks"])
            gc = h.array(frame["class_ids"]);raw = h.array(raw_t);fc = h.array(native["class_ids"])
            prob = h.array(output["class_logits"].softmax(-1));pc = prob.argmax(1)-1
            assignment = None
            if scope=="train_probe":
                labels,classes,_ = build_targets(frame,output["valid_points"]);labels = labels.clamp_min(0)
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(2026+index)
                    mi = sample_supervised_points(labels,output["valid_points"],model.config.match_points,model.config.points_per_instance)
                assignment = np.asarray(MaskPLSCriterion(model.config).match(output,labels,classes,mi))[:len(gc)]
            row,targets = h.score_arrays(h.array(gt_t),gc,raw,pc,h.array(final_t),fc,prob,roi,valid,full,pan,
                cpu["scene_token"],index,assignment,h.gpu_iou(gt_t,raw_t),h.gpu_iou(gt_t,final_t))
            enrich_targets(targets,h.array(gt_t),raw,h.array(frame["points"][:,:3])[valid])
            arrays = dict(valid=valid,roi=roi,gt_class=gc,query_class=pc,class_probability=prob,
                raw_packed=np.packbits(raw,axis=1,bitorder="little"),
                final_full_packed=np.packbits(full,axis=1,bitorder="little"),final_class=fc)
            if assignment is not None:arrays["matching_assignment"] = assignment
            row = h.save_frame(folder/f"frame-{index:04d}",arrays,row,targets,src)
            h.log(folder/"frames.jsonl",row);rows.append(row);all_targets.extend(targets)
            h.write(a.out/"progress.json",dict(stage="evaluation",identity=identity,scope=scope,completed=ordinal+1,expected=len(indices)))
            del output,frame,native
        if tensor_digest(model.state_dict())!=before:
            raise ValueError("evaluation changed model state")
    h.write(folder/"metrics.json",summary(rows,all_targets))
    h.write(folder/"completion.json",dict(complete=True,frames=len(rows),identity=dict(**identity,
        state_sha256=before,eval_state_unchanged=True)))
    return h.read(folder/"metrics.json")


def smoke(a):
    plan = h.read(a.out/"plan.json");check_sources(a,plan)
    if (a.out/"smoke.json").exists() or (a.out/"F").exists():
        raise ValueError("smoke output already exists; preserve it and use a new run")
    item = h.read(a.out/"sampling_manifest.json")[0]
    records = {};query_ids = {}
    for arm in ARMS:
        model,identity = new_model(a,arm)
        ds = h.datasets(a,model.config)
        frame = to_device(ds["train_probe"][item["index"]]["frames"][0],torch.device("cuda:0"))
        mi,li = h.samples(a,item,frame["points"].device)
        model.train();torch.cuda.reset_peak_memory_stats();start = time.monotonic()
        output = model(frame["points"])
        loss,parts,_ = MaskPLSCriterion(model.config)(output,frame,match_indices=mi,loss_indices=li)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite smoke loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        flags = identity["trainable"]
        for name,p in model.named_parameters():
            if p.requires_grad != flags[name] or (not flags[name] and p.grad is not None):
                raise ValueError("query gradient contract failed: "+name)
        for key in QUERY_KEYS:
            p = dict(model.named_parameters())[key]
            if flags[key] and (p.grad is None or not torch.isfinite(p.grad).all()):
                raise ValueError("active query table has no finite gradient")
        if not any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n not in QUERY_KEYS):
            raise ValueError("shared network has no nonzero gradient")
        if arm == "Q2":
            local = model.decoder.last_query_point_indices[0].numpy()
            roi_ids = output["point_indices"].cpu().numpy()
            query_ids = dict(roi_row_ids=local.tolist(),original_row_ids=roi_ids[local].tolist())
        records[arm] = dict(**identity,loss=float(loss.detach()),parts={k:float(v.detach()) for k,v in parts.items()},
            gradient_norm=float(norm),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            seconds=time.monotonic()-start,optimizer_updates=0)
        del model,frame,output,loss,parts;torch.cuda.empty_cache()
    if len({r["shared_state_sha256"] for r in records.values()}) != 1:
        raise ValueError("shared initial states differ")
    if records["Q0"]["query_sha256"][QUERY_KEYS[1]] != records["Q1"]["query_sha256"][QUERY_KEYS[1]]:
        raise ValueError("Q0/Q1 position initialization differs")
    # Reproduce F under the new source/hardware before ANY optimizer update.
    model,identity = new_model(a)
    if identity["initial_state_sha256"] != plan["parent_state_sha256"]:
        raise ValueError("frozen parent state mismatch")
    ds = h.datasets(a,model.config)
    for scope,indices in (("train_probe",plan["train_probe_indices"]),("dev",plan["dev_indices"])):
        evaluate(a,model,ds[scope],scope,indices,a.out/"F"/scope,
                 dict(kind="frozen_epoch14_reproduction",checkpoint_sha256=h.PARENT_SHA,epoch=14))
    actual = h.read(a.out/"F/dev/metrics.json");reference = plan["reference_F_dev"]
    # Lock a practical discrete-output continuity check before seeing any arm.
    # A changed baseline requires investigation and a new predeclared plan.
    for key in ("frames","gt","tp","pred","fp","raw_geometry_tp","raw_class_tp"):
        if actual[key] != reference[key]:
            raise ValueError("F baseline continuity failed for "+key)
    h.write(a.out/"baseline_reproduction.json",dict(complete=True,reference=reference,actual=actual,
        exact_discrete_totals=True,training_updates=0,new_forwards=160,
        note="Full strata recorded; matching tie allocation and float equality are not presumed across hardware"))
    h.write(a.out/"smoke.json",dict(complete=True,training_updates=0,extra_forwards=3,
        baseline_forwards=160,initializations=records,Q2_query_points=query_ids,baseline_reproduced=True))
    del model;torch.cuda.empty_cache();check_sources(a,plan)


def train_arm(a,arm):
    plan = h.read(a.out/"plan.json");check_sources(a,plan)
    smoke_record = h.read(a.out/"smoke.json")
    if not smoke_record["complete"]:
        raise ValueError("CUDA smoke is incomplete")
    folder = a.out/arm;folder.mkdir(exist_ok=False)
    model,initial = new_model(a,arm)
    if initial != {k:v for k,v in smoke_record["initializations"][arm].items() if k in initial}:
        raise ValueError("initialization differs from CUDA smoke")
    ds = h.datasets(a,model.config)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=80,gamma=.1)
    if optimizer.state:
        raise ValueError("optimizer was not reset")
    origin = plan["parent_metadata"]
    metadata = base_metadata(model.config)
    metadata.update(e2q_plan_sha256=h.sha(a.out/"plan.json"),arm=arm,recipe=plan["recipe"],
        train_manifest_sha256=plan["train_manifest_sha256"],dev_manifest_sha256=plan["dev_manifest_sha256"],
        data=origin["data"],internal_dev_data=origin["internal_dev_data"],
        initialization=dict(kind="explicit_e2q_model_only_migration",checkpoint_sha256=h.PARENT_SHA,
            source_implementation_sha256=h.PARENT_IMPLEMENTATION,optimizer_reset=True,scheduler_reset=True,**initial),
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    h.write(folder/"initialization.json",dict(**initial,optimizer_states=0,scheduler_last_epoch=scheduler.last_epoch))
    def assess(epoch,checkpoint_sha):
        for scope,indices in (("train_probe",plan["train_probe_indices"]),("dev",plan["dev_indices"])):
            evaluate(a,model,ds[scope],scope,indices,folder/f"eval-{epoch:02d}"/scope,
                     dict(arm=arm,epoch=epoch,global_step=epoch*191,checkpoint_sha256=checkpoint_sha))
        if {n:p.requires_grad for n,p in model.named_parameters()} != initial["trainable"]:
            raise ValueError("evaluation changed trainability")
    assess(0,None)
    manifest = h.read(a.out/"sampling_manifest.json");step = 0;streak = 0
    frozen = h.read(a.out/"F/dev/metrics.json")
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,7):
        model.train();configure_trainability(model,arm)
        items = [r for r in manifest if r["epoch"]==epoch]
        if len(items)!=764:
            raise ValueError("training frame count changed")
        for offset in range(0,764,4):
            optimizer.zero_grad(set_to_none=True);parts = Counter();start = time.monotonic()
            for item in items[offset:offset+4]:
                h.seed(402600000+(epoch-1)*764+item["ordinal"])
                frame = to_device(ds["train_probe"][item["index"]]["frames"][0],torch.device("cuda:0"))
                mi,li = h.samples(a,item,frame["points"].device)
                output = model(frame["points"])
                loss,components,_ = MaskPLSCriterion(model.config)(output,frame,match_indices=mi,loss_indices=li)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                (loss/4).backward()
                for key,value in components.items():parts[key]+=float(value.detach())/4
                h.log(folder/"training_frames.jsonl",dict(epoch=epoch,ordinal=item["ordinal"],dataset_index=item["index"],
                    sampling_sha256=item["sha256"],loss=float(loss.detach())))
                del frame,output,loss,components,mi,li
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            h.log(folder/"steps.jsonl",dict(arm=arm,epoch=epoch,global_step=step,gradient_norm=float(norm),
                  seconds=time.monotonic()-start,**parts))
            h.write(a.out/"progress.json",dict(stage="training",arm=arm,epoch=epoch,global_step=step,
                    expected_steps=1146,frame_exposures=(epoch-1)*764+offset+4))
            if step%10==0:print("TRAIN",arm,epoch,step,dict(parts),flush=True)
        optimizer.zero_grad(set_to_none=True);scheduler.step()
        if step!=epoch*191 or any(group["lr"]!=1e-4 for group in optimizer.param_groups):
            raise ValueError("training budget/learning rate mismatch")
        checkpoint = folder/f"epoch-{epoch:02d}.pt";temporary = checkpoint.with_suffix(".pt.part")
        torch.save(dict(metadata=metadata,model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),rng=rng_state(),completed_epoch=epoch,global_step=step),temporary)
        temporary.replace(checkpoint)
        if epoch in plan["evaluation_epochs"]:
            assess(epoch,h.sha(checkpoint))
            dev = h.read(folder/f"eval-{epoch:02d}/dev/metrics.json")
            bad = dev["tp"]<frozen["tp"]*.5 or dev["strata"]["bucket"]["1-10"]["raw_geometry_tp"]<frozen["strata"]["bucket"]["1-10"]["raw_geometry_tp"]*.5
            streak = streak+1 if bad else 0
            h.log(folder/"evaluations.jsonl",dict(epoch=epoch,global_step=step,dev=dev,
                catastrophic_streak=streak,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
            if streak>=2:
                h.write(folder/"completion.json",dict(complete=False,status="stopped_early",completed_epoch=epoch,
                    updates=step,frame_exposures=epoch*764,reason="two consecutive catastrophic prescribed evaluations"))
                break
    else:
        h.write(folder/"completion.json",dict(complete=True,status="completed",completed_epoch=6,updates=step,frame_exposures=4584))
    del model,optimizer,scheduler;torch.cuda.empty_cache();check_sources(a,plan)


def audit(a):
    plan = h.read(a.out/"plan.json");check_sources(a,plan)
    cfg = MaskPLSConfig.from_dict(plan["config"]);scopes = []
    folders = sorted(a.out.glob("Q*/eval-*/*"))+list((a.out/"F").glob("*"))
    for folder in folders:
        if not (folder/"frames.jsonl").exists():continue
        expected = plan["train_probe_indices"] if folder.name=="train_probe" else plan["dev_indices"]
        records = [json.loads(s) for s in (folder/"frames.jsonl").read_text().splitlines()]
        if [r["dataset_index"] for r in records]!=expected:
            raise ValueError("evaluation frame scope/order mismatch")
        checked = [];targets = []
        for row in records:
            src = row["source"];idx = row["dataset_index"];dest = folder/f"frame-{idx:04d}"
            tv = a.parent_root/"datageneration/data/trainval-64-001";panroot = a.data/"nuScenes-panoptic-v1.0-all"
            for base,key in ((a.data,"point_file"),(tv,"mask_file"),(panroot,"source_panoptic_file")):
                if h.sha(base/src[key]["path"])!=src[key]["sha256"]:
                    raise ValueError("source point/GT identity mismatch")
            xyz = np.fromfile(a.data/src["point_file"]["path"],dtype="<f4").reshape(-1,5)[:,:3]
            with np.load(tv/src["mask_file"]["path"]) as z:
                labels = z["labels"]==1
                if not np.array_equal(z["point_indices"],np.arange(len(xyz))):
                    raise ValueError("GT original point axis changed")
            with np.load(panroot/src["source_panoptic_file"]["path"]) as z:pan = z["data"].reshape(-1)
            bounds = np.asarray(cfg.point_cloud_range);roi = ((xyz>=bounds[:3])&(xyz<bounds[3:])).all(1)
            valid = roi&~np.isin(pan//1000,plan["ignored_raw_semantic_ids"])
            if h.sha(dest/"prediction.npz")!=row["snapshot_sha256"]:
                raise ValueError("saved prediction identity changed")
            with np.load(dest/"prediction.npz") as z:
                if not np.array_equal(roi,z["roi"]) or not np.array_equal(valid,z["valid"]) or not np.array_equal(z["gt_class"],src["class_ids"]):
                    raise ValueError("prediction/GT point scope mismatch")
                raw = np.unpackbits(z["raw_packed"],axis=1,count=int(valid.sum()),bitorder="little").astype(bool)
                full = np.unpackbits(z["final_full_packed"],axis=1,count=len(valid),bitorder="little").astype(bool)
                if full[:,~roi].any():raise ValueError("prediction escaped ROI")
                assigned = z["matching_assignment"] if "matching_assignment" in z else None
                new,selected = h.score_arrays(labels[:,valid],z["gt_class"],raw,z["query_class"],full[:,valid],z["final_class"],
                    z["class_probability"],roi,valid,full,pan,row["scene_token"],idx,assigned)
                enrich_targets(selected,labels[:,valid],raw,xyz[valid])
            if any(row[k]!=v for k,v in new.items()) or h.read(dest/"targets.json")!=selected:
                raise ValueError("independent CPU IoU/matching/strata disagreement")
            checked.append(new);targets.extend(selected)
        if summary(checked,targets)!=h.read(folder/"metrics.json"):
            raise ValueError("independent aggregate metrics disagree")
        scopes.append(dict(path=folder.relative_to(a.out).as_posix(),frames=len(checked),complete=True))
        print("AUDITED",folder,flush=True)
    done = {arm:h.read(a.out/arm/"completion.json") for arm in ARMS}
    expected_scopes = 2+sum(2*(1+d["completed_epoch"]//2) for d in done.values())
    if len(scopes)!=expected_scopes:raise ValueError("missing prescribed evaluation scopes")
    sampling = h.read(a.out/"sampling_manifest.json")
    initial = {arm:h.read(a.out/arm/"initialization.json") for arm in ARMS}
    if len({r["shared_state_sha256"] for r in initial.values()})!=1 or initial["Q0"]["query_sha256"][QUERY_KEYS[1]]!=initial["Q1"]["query_sha256"][QUERY_KEYS[1]]:
        raise ValueError("paired initialization audit failed")
    for scope in ("train_probe","dev"):
        identity = h.read(a.out/"F"/scope/"completion.json")["identity"]
        if identity["state_sha256"]!=plan["parent_state_sha256"]:
            raise ValueError("frozen baseline evaluated a different model state")
    parameter_reports = {}
    for arm,d in done.items():
        for scope in ("train_probe","dev"):
            identity = h.read(a.out/arm/"eval-00"/scope/"completion.json")["identity"]
            if identity["state_sha256"]!=initial[arm]["initial_state_sha256"]:
                raise ValueError("epoch-zero evaluation differs from paired initialization")
        rows = [json.loads(s) for s in (a.out/arm/"training_frames.jsonl").read_text().splitlines()]
        expected = sampling[:d["frame_exposures"]]
        if [(r["epoch"],r["ordinal"],r["dataset_index"],r["sampling_sha256"]) for r in rows]!=[(r["epoch"],r["ordinal"],r["index"],r["sha256"]) for r in expected]:
            raise ValueError("actual paired frame/sampling stream mismatch")
        steps = [json.loads(s) for s in (a.out/arm/"steps.jsonl").read_text().splitlines()]
        if [r["global_step"] for r in steps]!=list(range(1,d["updates"]+1)):
            raise ValueError("optimizer update log mismatch")
        for item in expected:
            if h.sha(a.out/item["relative_path"])!=item["sha256"]:
                raise ValueError("consumed sample file changed")
        for epoch in range(1,d["completed_epoch"]+1):
            path = a.out/arm/f"epoch-{epoch:02d}.pt";ckpt = load_native(path)
            if ckpt["global_step"]!=191*epoch or ckpt["metadata"]["e2q_plan_sha256"]!=h.sha(a.out/"plan.json"):
                raise ValueError("checkpoint identity/update mismatch")
            counters = [int(s["step"]) for s in ckpt["optimizer"]["state"].values()]
            if not counters or min(counters)!=max(counters) or max(counters)!=191*epoch:
                raise ValueError("Adam update counters disagree")
            active = [n for n,flag in initial[arm]["trainable"].items() if flag]
            groups = ckpt["optimizer"]["param_groups"]
            if len(groups)!=1 or len(groups[0]["params"])!=len(active):
                raise ValueError("optimizer parameter selection differs from arm contract")
            ids = dict(zip(active,groups[0]["params"]))
            unused = [n for n,idx in ids.items() if idx not in ckpt["optimizer"]["state"]]
            if set(unused)-{"decoder.level_embed.weight"}:
                raise ValueError("unexpected trainable parameter never received gradients: "+str(unused))
            if any(not torch.isfinite(v).all() for v in ckpt["model"].values() if v.is_floating_point()):
                raise ValueError("nonfinite checkpoint tensor")
            for key in QUERY_KEYS:
                if not initial[arm]["trainable"][key] and tensor_digest({key:ckpt["model"][key]})!=initial[arm]["query_sha256"][key]:
                    raise ValueError("a fixed query table changed")
            if epoch in plan["evaluation_epochs"]:
                for scope in ("train_probe","dev"):
                    identity = h.read(a.out/arm/f"eval-{epoch:02d}"/scope/"completion.json")["identity"]
                    if identity["checkpoint_sha256"]!=h.sha(path) or identity["state_sha256"]!=tensor_digest(ckpt["model"]):
                        raise ValueError("evaluated/checkpoint model identity differs")
            parameter_reports[arm] = dict(optimizer_state_tensors=len(counters),unused_native_parameters=unused,
                fixed_tables=[k for k in QUERY_KEYS if not initial[arm]["trainable"][k]])
            del ckpt
    h.write(a.out/"audit.json",dict(complete=True,scopes=scopes,total_evaluation_frames=sum(r["frames"] for r in scopes),
        independent_cpu_iou_and_matching_cardinality=True,all_source_point_axis_and_prediction_hashes=True,
        paired_initialization_sampling_order_updates_and_checkpoint_states_verified=True,parameter_reports=parameter_reports,parent_unchanged=True))


def finalize(a):
    plan = h.read(a.out/"plan.json");check_sources(a,plan)
    audit_record = h.read(a.out/"audit.json")
    if not audit_record["complete"]:raise ValueError("independent audit is incomplete")
    done = {arm:h.read(a.out/arm/"completion.json") for arm in ARMS}
    endpoint = all(d["complete"] and d["completed_epoch"]==6 for d in done.values())
    comparison = {};gates = {};q1_gates = {}
    if endpoint:
        comparison = {"F":h.read(a.out/"F/dev/metrics.json")}
        comparison.update({arm:h.read(a.out/arm/"eval-06/dev/metrics.json") for arm in ARMS})
        gates = selection_gates(comparison["Q2"],[comparison[k] for k in ("F","Q0","Q1")])
        q1_gates = selection_gates(comparison["Q1"],[comparison[k] for k in ("F","Q0")])
    passed = bool(gates) and all(gates.values())
    h.write(a.out/"comparison.json",dict(metrics=comparison,Q2_gates=gates,Q1_content_only_gates=q1_gates,
        screen_passed=passed,caveat=plan["scope"]))
    h.write(a.out/"completion.json",dict(complete=True,full_e2q_complete=endpoint,planned_endpoint_completed=endpoint,
        status="completed" if endpoint else "completed_with_early_failure",arms=done,
        actual_optimizer_updates=sum(d["updates"] for d in done.values()),
        actual_training_frame_exposures=sum(d["frame_exposures"] for d in done.values()),
        audited_evaluation_frames=audit_record["total_evaluation_frames"],extra_smoke_forwards=3,
        audit_complete=True,screen_passed=passed,candidate_adopted=False,e3_started=False,
        human_reviewed=False,new_holdout_evaluated=False,model_performance_accepted=False))
    lines = ["# E2-Q 执行报告","",f"固定第6轮终点完成：{endpoint}；工程筛选通过：{passed}。",
             "当前为已暴露开发集上的单种子适应实验；不代表最终性能或语言/时序验收。", "",
             "| 模型 | TP | FP | Precision | Recall | 小目标几何TP | 小目标最终TP |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name,m in comparison.items():
        b = m["strata"]["bucket"]["1-10"]
        lines.append(f"| {name} | {m['tp']} | {m['fp']} | {m['precision']:.2%} | {m['recall']:.2%} | {b['raw_geometry_tp']} | {b['tp']} |")
    lines.extend(["",f"Q2 对全部控制门槛：{json.dumps(gates,ensure_ascii=False)}", "",
                  "失败、提前终止和未完成的人工检查保留；不选择中途最好权重代替第6轮。",
                  "多种子与新留出集未自动启动，原epoch14基线未替换。"])
    (a.out/"E2-Q执行报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","smoke","run","audit","finalize"))
    parser.add_argument("--parent-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True,help="audited E2-C delivery root with F metrics")
    parser.add_argument("--data",type=Path,default=Path("/root/autodl-tmp/nuScenes"))
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--arm",choices=tuple(ARMS),help="run a single arm; default sequential Q0,Q1,Q2")
    a = parser.parse_args()
    if a.mode=="prepare" and a.out.exists():
        raise ValueError("prepare requires a NEW output directory; existing evidence was preserved")
    if not torch.backends.cuda.is_built() and a.mode in ("smoke","run"):
        raise RuntimeError("CUDA-enabled wqlc is required")
    try:
        if a.mode=="prepare":prepare(a)
        elif a.mode=="smoke":smoke(a)
        elif a.mode=="run":
            for arm in ([a.arm] if a.arm else ARMS):train_arm(a,arm)
        elif a.mode=="audit":audit(a)
        else:finalize(a)
    except Exception as error:
        a.out.mkdir(parents=True,exist_ok=True)
        h.write(a.out/"failure.json",dict(mode=a.mode,error_type=type(error).__name__,error=str(error)))
        raise


if __name__=="__main__":main()
