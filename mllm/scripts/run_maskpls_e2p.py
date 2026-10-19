"""E2-P: mature-query mask residual ablation, fixed six-epoch endpoint.

Reuse audited E2-Q data streams and metric helpers; preserve all old evidence.
The architecture migration is explicit and never bypasses load_native identity.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch

from grounded4d.train import to_device, rng_state
from maskpls_b4dl.config import MaskPLSConfig
from maskpls_b4dl.criterion import MaskPLSCriterion
from maskpls_b4dl.e2p import ARMS, initialize_model, selection_gates
from maskpls_b4dl.e2q import tensor_digest
from maskpls_b4dl.model import MaskPLS
from maskpls_b4dl.protocol import implementation_digest
from maskpls_b4dl.runner import base_metadata, load_native

spec = importlib.util.spec_from_file_location("e2p_metric_dependency",Path(__file__).with_name("run_maskpls_e2q.py"))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
h = q.h


def source_files():
    files = q.source_files()
    root = Path(__file__).resolve().parents[2]
    for path in (Path(__file__),Path(__file__).with_name("run_maskpls_e2p_workflow.sh")):
        files[path.relative_to(root).as_posix()] = h.sha(path)
    return files


def check_sources(a,plan):
    if source_files() != plan["source_files"] or implementation_digest() != plan["implementation_sha256"]:
        raise ValueError("E2-P deployed source identity changed")
    parent = a.parent_root/"training_logs/full64-budget-001/checkpoints/epoch-0014.pt"
    if h.sha(parent) != h.PARENT_SHA:
        raise ValueError("parent checkpoint changed")
    tv = a.parent_root/"datageneration/data/trainval-64-001"
    for name,key in (("train.jsonl","train_manifest_sha256"),("internal_dev.jsonl","dev_manifest_sha256")):
        if h.sha(tv/name) != plan[key]:raise ValueError("data manifest changed")
    for name,key in (("frame_order.npy","frame_order_sha256"),("sampling_manifest.json","sampling_manifest_sha256")):
        if h.sha(a.out/name) != plan[key]:raise ValueError("paired data stream changed")


def new_model(a,arm="P0"):
    _,base = q.parent_config(a)
    parent = h.parent(a,base)
    config = MaskPLSConfig.from_dict(dict(base.to_dict(),mask_residual_mode=ARMS[arm]))
    h.seed(2026)
    model = MaskPLS(config)
    identity = initialize_model(model,parent["model"],arm)
    del parent
    return model.cuda(),identity


def prepare(a):
    if a.out.exists():raise ValueError("prepare requires a NEW output directory")
    old,cfg = q.parent_config(a)
    parent = h.parent(a,cfg)
    reference = h.read(a.reference/"plan.json")
    done = h.read(a.reference/"completion.json")
    if not (done.get("full_e2q_complete") and done.get("audit_complete")
            and h.read(a.reference/"audit.json")["complete"]):
        raise ValueError("E2-Q reference audit is incomplete")
    for key in ("parent_checkpoint_sha256","train_manifest_sha256","dev_manifest_sha256"):
        expected = h.PARENT_SHA if key=="parent_checkpoint_sha256" else old[key]
        if reference[key]!=expected:raise ValueError("reference data or parent mismatch")
    if MaskPLSConfig.from_dict(reference["config"]).to_dict()!=cfg.to_dict():
        raise ValueError("reference recipe mismatch")
    if reference["parent_state_sha256"]!=tensor_digest(parent["model"]):
        raise ValueError("reference parent state mismatch")
    tv = a.parent_root/"datageneration/data/trainval-64-001"
    for name,key in (("train.jsonl","train_manifest_sha256"),("internal_dev.jsonl","dev_manifest_sha256")):
        if h.sha(tv/name)!=old[key]:raise ValueError("parent manifest mismatch")
    metadata = parent["metadata"];parent_digest = tensor_digest(parent["model"])
    del parent
    for data in (metadata["data"],metadata["internal_dev_data"]):
        for name,expected in data["source_split"]["metadata_sha256"].items():
            base = a.data/"nuScenes-panoptic-v1.0-all" if name=="panoptic" else a.data
            if h.sha(base/"v1.0-trainval"/(name+".json"))!=expected:
                raise ValueError("nuScenes metadata changed")
    reference_f = h.read(a.reference/"F/dev/metrics.json")
    if (reference_f["frames"],reference_f["gt"],reference_f["tp"])!=(96,3217,188):
        raise ValueError("unexpected frozen F reference")
    order_path = a.reference/"frame_order.npy"
    if h.sha(order_path)!=reference["frame_order_sha256"]:raise ValueError("reference order changed")
    order = np.load(order_path)
    expected_order = np.stack([np.random.default_rng(2026+e).permutation(764) for e in range(6)])
    if not np.array_equal(order,expected_order):raise ValueError("unexpected paired order")
    sampling_path = a.reference/"sampling_manifest.json"
    if h.sha(sampling_path)!=reference["sampling_manifest_sha256"]:
        raise ValueError("reference sample manifest changed")
    sampling = h.read(sampling_path)
    if [(r["epoch"],r["ordinal"],r["index"]) for r in sampling]!=[
            (e+1,i,int(index)) for e,row in enumerate(order) for i,index in enumerate(row)]:
        raise ValueError("reference sampling/order mismatch")
    a.out.mkdir(parents=True)
    shutil.copyfile(order_path,a.out/"frame_order.npy")
    shutil.copyfile(sampling_path,a.out/"sampling_manifest.json")
    # Copy bytes into the NEW run, so old evidence is never mutated by the run.
    for item in sampling:
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0]!="sampling":
            raise ValueError("invalid reference sampling path")
        src = a.reference/relative;dest = a.out/relative
        if h.sha(src)!=item["sha256"]:raise ValueError("reference sample changed")
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        if h.sha(dest)!=item["sha256"]:raise ValueError("sample copy changed")
    # Re-read all actual records: original point/GT/panoptic checks are mandatory.
    datasets = h.datasets(a,cfg)
    for scope in ("train_probe","dev"):
        for i in range(len(datasets[scope])):
            datasets[scope][i]
            if i%100==0:print("PREPARE",scope,i+1,flush=True)
    plan = dict(experiment="E2-P",schema="b4dl-maskpls-e2p-v1",source_commit=a.source_commit,
        source_files=source_files(),implementation_sha256=implementation_digest(),
        parent_checkpoint_sha256=h.PARENT_SHA,parent_implementation_sha256=h.PARENT_IMPLEMENTATION,
        parent_state_sha256=parent_digest,parent_metadata=metadata,config=cfg.to_dict(),arms=ARMS,
        reset_parameters=[],new_parameters="P1/P2: decoder.mask_residual_embed.layers.{0,1,2}.{weight,bias}",
        train_manifest_sha256=old["train_manifest_sha256"],dev_manifest_sha256=old["dev_manifest_sha256"],
        train_probe_indices=old["train_probe_indices"],dev_indices=list(range(96)),
        ignored_raw_semantic_ids=old["ignored_raw_semantic_ids"],
        frame_order_sha256=h.sha(a.out/"frame_order.npy"),sampling_manifest_sha256=h.sha(a.out/"sampling_manifest.json"),
        reference_plan_sha256=h.sha(a.reference/"plan.json"),reference_completion_sha256=h.sha(a.reference/"completion.json"),
        reference_F_dev=reference_f,epochs_per_arm=6,updates_per_epoch=191,updates_per_arm=1146,
        total_planned_updates=3438,frame_exposures_per_arm=4584,total_planned_frame_exposures=13752,
        evaluation_epochs=[0,2,4,6],primary_endpoint=6,planned_evaluation_forwards=2080,
        recipe=dict(seed=2026,lr=1e-4,weight_decay=.01,batch_size=1,accumulate=4,gradient_clip=1.,
                    precision="FP32",scheduler="constant 1e-4 for six epochs",trainable="all shared and new parameters"),
        mechanism=dict(base="unchanged dot(mask_embed(norm(query)), feature + existing PE)",
            P0="base",P1="base + dot(zero-final 3-layer MLP(norm(query)), feature)",
            P2="base + dot(same MLP(norm(query)), existing PE(raw XYZ))",
            heads="initial and all 9 decoder heads; same auxiliary losses and subsequent attention masks",
            initialization="all F tensors including mature queries retained exactly; paired private CPU head seed; final weight and bias zero",
            input="raw LiDAR XYZ/intensity only; no GT/camera/answer/target centers"),
        gates=dict(final_tp_margin=10,precision_absolute_drop=.02,small_final_tp_margin=10,
            small_geometry_tp_margin=20,small_median_purity_margin=.01,small_median_coverage_drop=.05,
            small_median_best_iou_drop=0,medium_large_recall_drop=.02,min_scenes_nondecreasing=5,
            ignored_fraction_increase=.01,ego_multiplier=1.1,ego_add=100),
        stop="nonfinite/identity/point-axis/leak immediately; eval2/4/6 twice consecutively below half F TP or small geometry TP",
        scope="single seed; exposed internal dev; custom P3Former-inspired residual ablation, not full P3Former; no language/temporal/new-holdout acceptance")
    h.write(a.out/"plan.json",plan);h.write(a.out/"preregistration.json",plan)
    h.write(a.out/"preparation_completion.json",dict(complete=True,training_updates=0,
        train_frames=764,dev_frames=96,sampling_files=len(sampling),plan_sha256=h.sha(a.out/"plan.json")))


def summary(rows,targets):
    result = q.summary(rows,targets)
    small = [t for t in targets if t["points"]<=10]
    if small:
        result["geometry_diagnostics"]["1-10"] = dict(gt=len(small),**{
            "median_"+key:float(np.median([t[key] for t in small]))
            for key in ("best_query_coverage","best_query_purity","best_iou")})
    return result


def evaluate(a,model,ds,scope,indices,folder,identity):
    q.evaluate(a,model,ds,scope,indices,folder,identity)
    rows = [json.loads(s) for s in (folder/"frames.jsonl").read_text().splitlines()]
    targets = [t for i in indices for t in h.read(folder/f"frame-{i:04d}/targets.json")]
    result = summary(rows,targets);h.write(folder/"metrics.json",result)
    return result


def checkpoint_roundtrip(a,model,optimizer,arm):
    """One disposable smoke update, then strict weight/Adam reload to a fresh model."""
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    with tempfile.TemporaryDirectory(prefix="smoke-",dir=a.out) as directory:
        path = Path(directory)/"roundtrip.pt"
        torch.save(dict(metadata=base_metadata(model.config),model=model.state_dict(),
                        optimizer=optimizer.state_dict(),global_step=1),path)
        saved = load_native(path)
        restored = MaskPLS(model.config).cuda();restored.load_state_dict(saved["model"],strict=True)
        opt = torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=.01)
        opt.load_state_dict(saved["optimizer"])
        if tensor_digest(restored.state_dict())!=tensor_digest(model.state_dict()):
            raise ValueError("smoke checkpoint roundtrip changed weights")
        # Verify that restoring the optimizer reproduces the NEXT update, not
        # merely its counters. Same synthetic gradients avoid an extra forward.
        for left,right in zip(model.parameters(),restored.parameters()):
            if left.requires_grad:
                left.grad = torch.full_like(left,.001);right.grad = torch.full_like(right,.001)
        optimizer.step();opt.step()
        if tensor_digest(restored.state_dict())!=tensor_digest(model.state_dict()):
            raise ValueError("restored optimizer next update differs")
        result = dict(strict_weight_reload=True,adam_next_update_equal=True,
                      checkpoint_sha256=h.sha(path),discarded_smoke_optimizer_updates=2)
        optimizer.zero_grad(set_to_none=True)
        del restored,opt,saved
    return result


def smoke(a):
    plan = h.read(a.out/"plan.json");check_sources(a,plan)
    if (a.out/"smoke.json").exists() or (a.out/"F").exists():raise ValueError("smoke evidence already exists")
    item = h.read(a.out/"sampling_manifest.json")[0];records = {}
    baseline_output = None
    for arm in ARMS:
        model,initial = new_model(a,arm);datasets = h.datasets(a,model.config)
        frame = to_device(datasets["train_probe"][item["index"]]["frames"][0],torch.device("cuda:0"))
        model.eval()
        with torch.no_grad():
            output = model(frame["points"])
            compare = {k:output[k] for k in ("class_logits","mask_logits","queries")}
            compare.update({f"aux_{i}_{k}":v[k] for i,v in enumerate(output["auxiliary"])
                            for k in ("class_logits","mask_logits")})
            compare = tensor_digest(compare)
            if baseline_output is None:baseline_output = compare
            elif compare!=baseline_output:
                raise ValueError("zero residual branch failed exact baseline equivalence")
        del output,compare
        model.train();torch.cuda.reset_peak_memory_stats();start = time.monotonic()
        mi,li = h.samples(a,item,frame["points"].device);output = model(frame["points"])
        loss,parts,_ = MaskPLSCriterion(model.config)(output,frame,match_indices=mi,loss_indices=li)
        if not torch.isfinite(loss):raise FloatingPointError("nonfinite smoke loss")
        loss.backward();norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if arm!="P0":
            for p in model.decoder.mask_residual_embed.layers[-1].parameters():
                if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum()>0:
                    raise ValueError("new residual final layer has no finite nonzero gradient")
        record = dict(**initial,loss=float(loss.detach()),gradient_norm=float(norm),
            parts={k:float(v.detach()) for k,v in parts.items()},peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            exact_baseline_equivalence=True,initial_outputs_sha256=baseline_output,seconds=time.monotonic()-start)
        del output,loss,parts,frame,mi,li
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        record["checkpoint_roundtrip"] = checkpoint_roundtrip(a,model,optimizer,arm)
        records[arm] = record;del model,optimizer;torch.cuda.empty_cache()
    if len({r["inherited_state_sha256"] for r in records.values()})!=1:
        raise ValueError("inherited F states differ")
    if (records["P1"]["added_state_sha256"]!=records["P2"]["added_state_sha256"]
            or records["P1"]["added_parameter_count"]!=records["P2"]["added_parameter_count"]):
        raise ValueError("new head initializations/capacities differ")
    model,initial = new_model(a)
    if initial["initial_state_sha256"]!=plan["parent_state_sha256"]:raise ValueError("F state mismatch")
    datasets = h.datasets(a,model.config)
    for scope,indices in (("train_probe",plan["train_probe_indices"]),("dev",plan["dev_indices"])):
        evaluate(a,model,datasets[scope],scope,indices,a.out/"F"/scope,
                 dict(kind="frozen_epoch14_reproduction",checkpoint_sha256=h.PARENT_SHA,epoch=14))
    actual = h.read(a.out/"F/dev/metrics.json")
    for key in ("frames","gt","tp","pred","fp","raw_geometry_tp","raw_class_tp"):
        if actual[key]!=plan["reference_F_dev"][key]:raise ValueError("F baseline continuity failed: "+key)
    h.write(a.out/"smoke.json",dict(complete=True,initializations=records,training_updates=0,
        scratch_optimizer_updates=6,extra_smoke_forwards=6,baseline_forwards=160,baseline_reproduced=True,
        note="scratch updates discarded; all arms reinitialized from F for actual training"))
    del model;torch.cuda.empty_cache();check_sources(a,plan)


def train_arm(a,arm):
    plan = h.read(a.out/"plan.json");check_sources(a,plan)
    smoke_record = h.read(a.out/"smoke.json")
    if not smoke_record["complete"]:raise ValueError("CUDA smoke incomplete")
    folder = a.out/arm;folder.mkdir(exist_ok=False)
    model,initial = new_model(a,arm)
    if initial!={k:v for k,v in smoke_record["initializations"][arm].items() if k in initial}:
        raise ValueError("initialization differs from smoke")
    datasets = h.datasets(a,model.config)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=80,gamma=.1)
    if optimizer.state:raise ValueError("optimizer not reset")
    origin = plan["parent_metadata"];metadata = base_metadata(model.config)
    metadata.update(e2p_plan_sha256=h.sha(a.out/"plan.json"),arm=arm,recipe=plan["recipe"],
        train_manifest_sha256=plan["train_manifest_sha256"],dev_manifest_sha256=plan["dev_manifest_sha256"],
        data=origin["data"],internal_dev_data=origin["internal_dev_data"],
        initialization=dict(kind="explicit_e2p_model_only_migration",checkpoint_sha256=h.PARENT_SHA,
            source_implementation_sha256=h.PARENT_IMPLEMENTATION,optimizer_reset=True,scheduler_reset=True,**initial),
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    h.write(folder/"initialization.json",dict(**initial,optimizer_states=0))
    def assess(epoch,checkpoint_sha):
        for scope,indices in (("train_probe",plan["train_probe_indices"]),("dev",plan["dev_indices"])):
            evaluate(a,model,datasets[scope],scope,indices,folder/f"eval-{epoch:02d}"/scope,
                     dict(arm=arm,epoch=epoch,global_step=epoch*191,checkpoint_sha256=checkpoint_sha))
    assess(0,None)
    manifest = h.read(a.out/"sampling_manifest.json");step = 0;streak = 0
    frozen = h.read(a.out/"F/dev/metrics.json");torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,7):
        model.train();items = [r for r in manifest if r["epoch"]==epoch]
        if len(items)!=764:raise ValueError("training frame count changed")
        for offset in range(0,764,4):
            optimizer.zero_grad(set_to_none=True);parts = Counter();start = time.monotonic()
            for item in items[offset:offset+4]:
                h.seed(402600000+(epoch-1)*764+item["ordinal"])
                frame = to_device(datasets["train_probe"][item["index"]]["frames"][0],torch.device("cuda:0"))
                mi,li = h.samples(a,item,frame["points"].device);output = model(frame["points"])
                loss,components,_ = MaskPLSCriterion(model.config)(output,frame,match_indices=mi,loss_indices=li)
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite training loss")
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
        if step!=epoch*191 or any(g["lr"]!=1e-4 for g in optimizer.param_groups):
            raise ValueError("training budget/lr mismatch")
        path = folder/f"epoch-{epoch:02d}.pt";temp = path.with_suffix(".pt.part")
        torch.save(dict(metadata=metadata,model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),rng=rng_state(),completed_epoch=epoch,global_step=step),temp)
        temp.replace(path)
        if epoch in plan["evaluation_epochs"]:
            assess(epoch,h.sha(path));dev = h.read(folder/f"eval-{epoch:02d}/dev/metrics.json")
            bad = dev["tp"]<frozen["tp"]*.5 or dev["strata"]["bucket"]["1-10"]["raw_geometry_tp"]<frozen["strata"]["bucket"]["1-10"]["raw_geometry_tp"]*.5
            streak = streak+1 if bad else 0
            h.log(folder/"evaluations.jsonl",dict(epoch=epoch,global_step=step,dev=dev,
                catastrophic_streak=streak,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
            if streak>=2:
                h.write(folder/"completion.json",dict(complete=False,status="stopped_early",completed_epoch=epoch,
                    updates=step,frame_exposures=epoch*764,reason="two consecutive catastrophic evaluations"))
                break
    else:
        h.write(folder/"completion.json",dict(complete=True,status="completed",completed_epoch=6,updates=step,frame_exposures=4584))
    del model,optimizer,scheduler;torch.cuda.empty_cache();check_sources(a,plan)


def audit_scope(a,plan,folder):
    cfg = MaskPLSConfig.from_dict(plan["config"])
    expected = plan["train_probe_indices"] if folder.name=="train_probe" else plan["dev_indices"]
    rows = [json.loads(s) for s in (folder/"frames.jsonl").read_text().splitlines()]
    if [r["dataset_index"] for r in rows]!=expected:raise ValueError("evaluation scope/order mismatch")
    checked = [];targets = []
    for row in rows:
        src = row["source"];idx = row["dataset_index"];dest = folder/f"frame-{idx:04d}"
        tv = a.parent_root/"datageneration/data/trainval-64-001";panroot = a.data/"nuScenes-panoptic-v1.0-all"
        for base,key in ((a.data,"point_file"),(tv,"mask_file"),(panroot,"source_panoptic_file")):
            if h.sha(base/src[key]["path"])!=src[key]["sha256"]:raise ValueError("source point/GT identity mismatch")
        xyz = np.fromfile(a.data/src["point_file"]["path"],dtype="<f4").reshape(-1,5)[:,:3]
        with np.load(tv/src["mask_file"]["path"]) as z:
            labels = z["labels"]==1
            if not np.array_equal(z["point_indices"],np.arange(len(xyz))):raise ValueError("original point axis changed")
        with np.load(panroot/src["source_panoptic_file"]["path"]) as z:pan = z["data"].reshape(-1)
        bounds = np.asarray(cfg.point_cloud_range);roi = ((xyz>=bounds[:3])&(xyz<bounds[3:])).all(1)
        valid = roi&~np.isin(pan//1000,plan["ignored_raw_semantic_ids"])
        if h.sha(dest/"prediction.npz")!=row["snapshot_sha256"]:raise ValueError("prediction identity changed")
        with np.load(dest/"prediction.npz") as z:
            if not (np.array_equal(roi,z["roi"]) and np.array_equal(valid,z["valid"])
                    and np.array_equal(z["gt_class"],src["class_ids"])):raise ValueError("prediction/GT point scope mismatch")
            raw = np.unpackbits(z["raw_packed"],axis=1,count=int(valid.sum()),bitorder="little").astype(bool)
            full = np.unpackbits(z["final_full_packed"],axis=1,count=len(valid),bitorder="little").astype(bool)
            if full[:,~roi].any():raise ValueError("prediction escaped ROI")
            assigned = z["matching_assignment"] if "matching_assignment" in z else None
            new,selected = h.score_arrays(labels[:,valid],z["gt_class"],raw,z["query_class"],full[:,valid],z["final_class"],
                z["class_probability"],roi,valid,full,pan,row["scene_token"],idx,assigned)
            q.enrich_targets(selected,labels[:,valid],raw,xyz[valid])
        if any(row[k]!=v for k,v in new.items()) or h.read(dest/"targets.json")!=selected:
            raise ValueError("independent CPU IoU/matching/strata disagreement")
        checked.append(new);targets.extend(selected)
    if summary(checked,targets)!=h.read(folder/"metrics.json"):raise ValueError("aggregate metrics disagree")
    print("AUDITED",folder,flush=True)
    return dict(path=folder.relative_to(a.out).as_posix(),frames=len(checked),complete=True)


def audit(a):
    plan = h.read(a.out/"plan.json");check_sources(a,plan)
    done = {arm:h.read(a.out/arm/"completion.json") for arm in ARMS}
    # Enumerate required paths explicitly: a missing folder cannot silently pass.
    folders = [a.out/"F"/s for s in ("train_probe","dev")]
    folders += [a.out/arm/f"eval-{e:02d}"/s for arm,d in done.items()
                for e in plan["evaluation_epochs"] if e<=d["completed_epoch"] for s in ("train_probe","dev")]
    scopes = [audit_scope(a,plan,f) for f in folders]
    initial = {arm:h.read(a.out/arm/"initialization.json") for arm in ARMS}
    if any(r["inherited_state_sha256"]!=plan["parent_state_sha256"] for r in initial.values()):
        raise ValueError("inherited F state audit failed")
    if initial["P1"]["added_state_sha256"]!=initial["P2"]["added_state_sha256"]:
        raise ValueError("paired new head audit failed")
    for s in ("train_probe","dev"):
        if h.read(a.out/"F"/s/"completion.json")["identity"]["state_sha256"]!=plan["parent_state_sha256"]:
            raise ValueError("F evaluated a different state")
    sampling = h.read(a.out/"sampling_manifest.json")
    for item in sampling:
        if h.sha(a.out/item["relative_path"])!=item["sha256"]:raise ValueError("sample bytes changed")
    parameter_reports = {}
    for arm,d in done.items():
        for s in ("train_probe","dev"):
            if h.read(a.out/arm/"eval-00"/s/"completion.json")["identity"]["state_sha256"]!=initial[arm]["initial_state_sha256"]:
                raise ValueError("epoch0 state mismatch")
        rows = [json.loads(s) for s in (a.out/arm/"training_frames.jsonl").read_text().splitlines()]
        expected = sampling[:d["frame_exposures"]]
        if [(r["epoch"],r["ordinal"],r["dataset_index"],r["sampling_sha256"]) for r in rows]!=[
                (r["epoch"],r["ordinal"],r["index"],r["sha256"]) for r in expected]:raise ValueError("actual paired frame stream mismatch")
        steps = [json.loads(s) for s in (a.out/arm/"steps.jsonl").read_text().splitlines()]
        if [r["global_step"] for r in steps]!=list(range(1,d["updates"]+1)):raise ValueError("update log mismatch")
        for epoch in range(1,d["completed_epoch"]+1):
            path = a.out/arm/f"epoch-{epoch:02d}.pt";ckpt = load_native(path)
            if (ckpt["global_step"]!=191*epoch or ckpt["metadata"]["e2p_plan_sha256"]!=h.sha(a.out/"plan.json")
                    or ckpt["metadata"]["config"]["mask_residual_mode"]!=ARMS[arm]):raise ValueError("checkpoint identity mismatch")
            states = ckpt["optimizer"]["state"];counters = [int(s["step"]) for s in states.values()]
            if not counters or min(counters)!=max(counters) or max(counters)!=191*epoch:raise ValueError("Adam counters disagree")
            active = [n for n,flag in initial[arm]["trainable"].items() if flag];groups = ckpt["optimizer"]["param_groups"]
            if len(groups)!=1 or len(groups[0]["params"])!=len(active):raise ValueError("optimizer parameter selection mismatch")
            unused = [n for n,idx in zip(active,groups[0]["params"]) if idx not in states]
            if set(unused)-{"decoder.level_embed.weight"}:raise ValueError("unexpected parameters without gradients: "+str(unused))
            if any(not torch.isfinite(v).all() for v in ckpt["model"].values() if v.is_floating_point()):raise ValueError("nonfinite checkpoint")
            if epoch in plan["evaluation_epochs"]:
                for scope in ("train_probe","dev"):
                    identity = h.read(a.out/arm/f"eval-{epoch:02d}"/scope/"completion.json")["identity"]
                    if identity["checkpoint_sha256"]!=h.sha(path) or identity["state_sha256"]!=tensor_digest(ckpt["model"]):
                        raise ValueError("evaluated checkpoint state mismatch")
            parameter_reports[arm] = dict(optimizer_state_tensors=len(counters),unused_native_parameters=unused)
            del ckpt
    h.write(a.out/"audit.json",dict(complete=True,scopes=scopes,total_evaluation_frames=sum(r["frames"] for r in scopes),
        independent_cpu_iou_and_matching_cardinality=True,all_source_point_axis_and_prediction_hashes=True,
        paired_initialization_sampling_order_updates_and_checkpoint_states_verified=True,parameter_reports=parameter_reports,parent_unchanged=True))


def finalize(a):
    plan = h.read(a.out/"plan.json");check_sources(a,plan);audited = h.read(a.out/"audit.json")
    if not audited["complete"]:raise ValueError("independent audit incomplete")
    done = {arm:h.read(a.out/arm/"completion.json") for arm in ARMS}
    endpoint = all(d["complete"] and d["completed_epoch"]==6 for d in done.values())
    comparison = {};gates = {}
    if endpoint:
        comparison = {"F":h.read(a.out/"F/dev/metrics.json")}
        comparison.update({arm:h.read(a.out/arm/"eval-06/dev/metrics.json") for arm in ARMS})
        gates = selection_gates(comparison["P2"],[comparison[k] for k in ("F","P0","P1")])
    passed = bool(gates) and all(gates.values())
    h.write(a.out/"comparison.json",dict(metrics=comparison,P2_gates=gates,screen_passed=passed,caveat=plan["scope"]))
    h.write(a.out/"completion.json",dict(complete=True,full_e2p_complete=endpoint,planned_endpoint_completed=endpoint,
        status="completed" if endpoint else "completed_with_early_failure",arms=done,
        actual_optimizer_updates=sum(d["updates"] for d in done.values()),
        actual_training_frame_exposures=sum(d["frame_exposures"] for d in done.values()),
        audited_evaluation_frames=audited["total_evaluation_frames"],scratch_optimizer_updates=6,
        extra_smoke_forwards=6,audit_complete=True,screen_passed=passed,candidate_adopted=False,e3_started=False,
        human_reviewed=False,new_holdout_evaluated=False,model_performance_accepted=False))
    lines = ["# E2-P 执行报告","",f"固定第6轮终点完成：{endpoint}；位置分支筛选通过：{passed}。",
        "", "| 模型 | TP | FP | Precision | 小目标几何TP | 小目标最终TP | 小目标纯度中位数 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name,m in comparison.items():
        b = m["strata"]["bucket"]["1-10"];diag = m["geometry_diagnostics"]["1-10"]
        lines.append(f"| {name} | {m['tp']} | {m['fp']} | {m['precision']:.2%} | {b['raw_geometry_tp']} | {b['tp']} | {diag['median_best_query_purity']:.5f} |")
    lines += ["",f"P2 对 F/P0/P1 门槛：{json.dumps(gates,ensure_ascii=False)}","",
        "保留全部失败及提前停止，不挑中途最好权重。自定义残差消融不等于复现P3Former。",
        "单种子与已暴露开发集结果不能替代多种子、人工检查和真正留出验收；原epoch14未替换，E3未启动。"]
    (a.out/"E2-P执行报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","smoke","run","audit","finalize"))
    parser.add_argument("--parent-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True,help="completed audited E2-Q run")
    parser.add_argument("--data",type=Path,default=Path("/root/autodl-tmp/nuScenes"))
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--arm",choices=tuple(ARMS))
    a = parser.parse_args()
    if a.mode=="prepare" and a.out.exists():raise ValueError("prepare requires a NEW output directory; evidence preserved")
    if a.mode in ("smoke","run") and not torch.backends.cuda.is_built():raise RuntimeError("CUDA wqlc required")
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
