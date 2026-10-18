"""Read-only fixed-weight query/competition trace, preserving native extraction.

Run in wqlc against an isolated, hash-verified E2-C source tree. No training.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from maskpls_b4dl.config import MaskPLSConfig, NO_OBJECT
from maskpls_b4dl.data import semantic_from_panoptic
from maskpls_b4dl.model import MaskPLS, extract_instances, sha256_file
from maskpls_b4dl.protocol import implementation_digest
from maskpls_b4dl.runner import load_native, load_warm_start
from run_maskpls_e2c import cpu_iou, digest, pairs, seed, write, array


@torch.no_grad()
def trace_extraction(output, cfg):
    """Independent trace; assert final ids, classes, and masks equal native output."""
    native = extract_instances(output, cfg)
    prob = output['class_logits'].float().softmax(-1)
    score, cls = prob.max(-1)
    mp = output['mask_logits'].float().sigmoid()
    roi = output['valid_points']
    eligible = (cls > 0) & (cls < NO_OBJECT) & (score >= cfg.score_threshold)
    ids = eligible.nonzero(as_tuple=True)[0]
    winner = torch.full((len(roi),), -1, device=roi.device, dtype=torch.long)
    if len(ids):
        winner = ids[(mp[ids] * score[ids, None]).argmax(0)]
    pre = mp.ge(cfg.mask_threshold) & roi[None]
    post = pre & winner[None].eq(torch.arange(len(cls), device=cls.device)[:, None])
    retained, reasons, rows = [], [], []
    for q in range(len(cls)):
        n, k = int(pre[q].sum()), int(post[q].sum())
        ratio = float(post[q].sum() / pre[q].sum().clamp_min(1))
        if cls[q] == NO_OBJECT:
            reason = 'no_object'
        elif cls[q] == 0:
            reason = 'ignore_class'
        elif score[q] < cfg.score_threshold:
            reason = 'below_score_threshold'
        elif cls[q] > 10:
            reason = 'stuff_competitor_not_exported'
        elif k == 0:
            reason = 'empty_after_competition' if n else 'empty_before_competition'
        elif ratio < cfg.overlap_threshold:
            reason = 'below_overlap_threshold'
        else:
            reason = 'retained'
            retained.append(q)
        reasons.append(reason)
        lost = pre[q] & ~post[q]
        lost_to = Counter(array(winner[lost]).tolist()) if eligible[q] else {}
        rows.append(dict(query=q, class_id=int(cls[q])-1, class_probability=float(score[q]),
                         no_object_probability=float(prob[q, NO_OBJECT]), eligible=bool(eligible[q]),
                         pre_roi_points=n, post_roi_points=k, retention_ratio=ratio,
                         reason=reason, lost_to_queries={str(a): b for a, b in lost_to.items()}))
    assert retained == native['query_indices'].tolist()
    assert torch.equal(post[retained], native['masks'])
    assert torch.equal(cls[retained]-1, native['class_ids'])
    return native, dict(class_probability=array(prob), mask_probability=array(mp),
                        pre=array(pre), post=array(post), winner_query=array(winner), rows=rows)


def unpack(z, key, n):
    return np.unpackbits(z[key], axis=1, count=n, bitorder='little').astype(bool)


def describe(mask, gt, gc, pan, valid):
    m = mask & valid
    inter = (gt & m[None]).sum(1)
    gt_size = gt.sum(1)
    union = gt_size + m.sum() - inter
    iou = np.divide(inter, union, out=np.zeros(len(gt), dtype=float), where=union > 0)
    overlap = [dict(gt=int(g), class_id=int(gc[g]), intersection=int(inter[g]),
                    gt_points=int(gt_size[g]), coverage=float(inter[g]/gt_size[g]),
                    iou=float(iou[g])) for g in np.flatnonzero(inter)]
    return dict(valid_points=int(m.sum()), ignored_points=int((mask & ~valid).sum()),
                semantic_fine_counts={str(k): int(v) for k, v in Counter((pan[m]//1000).tolist()).items()},
                thing_union_points=int((m & gt.any(0)).sum()), overlaps=overlap)


def verified_inputs(review):
    manifest = json.loads((review/'file_manifest.json').read_text(encoding='utf-8-sig'))
    checks = []
    for row in manifest['files']:
        name = row['path']
        p = review/name
        # Images and editable documentation are not inference inputs.
        if name in ('cases.json', 'frames.json') or (name.startswith('frames/') and not name.endswith('.png')):
            assert p.resolve().is_relative_to(review.resolve())
            assert sha256_file(p) == row['sha256'], name
            checks.append(row)
    for f in (44, 56, 62):
        d = review/f'frames/frame-{f:04d}'
        source = json.loads((d/'identity.json').read_text(encoding='utf-8-sig'))['source']
        assert sha256_file(d/'points.pcd.bin') == source['point_file']['sha256']
        assert sha256_file(d/'gt_masks.npz') == source['mask_file']['sha256']
        assert sha256_file(d/'panoptic.npz') == source['source_panoptic_file']['sha256']
        assert np.allclose(source['lidar_to_anchor'], np.eye(4))
    return checks


@torch.no_grad()
def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(a.plan.read_text(encoding='utf-8-sig'))
    root = Path(__file__).resolve().parents[2]
    for name, expected in plan['source_files'].items():
        assert sha256_file(root/name) == expected, name
    assert implementation_digest() == plan['implementation_sha256']
    inputs = verified_inputs(a.review)
    expected = dict(F=plan['parent_checkpoint_sha256'],
                    C0='0e6656bad78f4319151695c231f99e996d49b0fcb828b83f5add6a0de3c38d65',
                    C1='305dfd56217e91e8a0037ca5f19c171f9868bcaa74a04a0c47c7633b5810ee9c')
    paths = dict(F=a.parent, C0=a.weights/'C0-epoch-03.pt', C1=a.weights/'C1-epoch-03.pt')
    for arm in a.arms:
        p = paths[arm]
        assert sha256_file(p) == expected[arm], arm
    cases = json.loads((a.review/'cases.json').read_text(encoding='utf-8-sig'))
    write(a.out/'inputs.json', dict(files=inputs, checkpoint_sha256=expected,
          source_implementation_sha256=implementation_digest(), script_sha256=sha256_file(__file__),
          device=torch.cuda.get_device_name(), torch_version=str(torch.__version__), training_updates=0))
    summaries, reviews, forwards = [], [], 0
    for arm in a.arms:
        seed(2026)
        cfg = MaskPLSConfig.from_dict(dict(plan['config'], no_object_weight=.05 if arm == 'C1' else .1))
        state = load_warm_start(paths[arm], cfg, plan['parent_implementation_sha256']) if arm == 'F' else load_native(paths[arm])
        if arm != 'F':
            assert state['metadata']['arm'] == arm and state['completed_epoch'] == 3
            assert state['metadata']['config'] == cfg.to_dict()
        model = MaskPLS(cfg).cuda().eval().requires_grad_(False)
        model.load_state_dict(state['model'], strict=True)
        del state
        before = digest(model)
        for f in (44, 56, 62):
            start = time.monotonic()
            d = a.review/f'frames/frame-{f:04d}'
            dest = a.out/arm/f'frame-{f:04d}'
            dest.mkdir(parents=True)
            points = np.fromfile(d/'points.pcd.bin', dtype=np.float32).reshape(-1, 5)[:, :4].copy()
            with np.load(d/'panoptic.npz') as z:
                pan = z['data'].reshape(-1)
            with np.load(d/'gt_masks.npz') as z:
                labels = z['labels'] == 1
            with np.load(d/f'{arm}-prediction.npz') as z:
                ref = {k:z[k].copy() for k in z.files}
            torch.cuda.reset_peak_memory_stats()
            output = model(torch.from_numpy(points).cuda())
            forwards += 1
            native, trace = trace_extraction(output, cfg)
            valid = array(output['valid_points']) & semantic_from_panoptic(pan).numpy().astype(bool)
            assert np.array_equal(valid, ref['valid'])
            assert np.array_equal(array(output['valid_points']), ref['roi'])
            gt, gc = labels[:, valid], ref['gt_class']
            full_gt = labels & valid[None]
            ids, full, fc = array(native['query_indices']), array(native['masks']), array(native['class_ids'])
            old = unpack(ref, 'final_full_packed', len(valid))
            old_raw = unpack(ref, 'raw_packed', int(valid.sum()))
            exact = np.array_equal(full, old) and np.array_equal(fc, ref['final_class'])
            raw_diff = int(np.count_nonzero(trace['pre'][:, valid] != old_raw))
            current_iou = cpu_iou(gt, full[:, valid])
            matched = pairs(current_iou, gc, fc)
            np.savez_compressed(dest/'trace.npz', class_probability=trace['class_probability'],
                mask_probability=trace['mask_probability'], roi=array(output['valid_points']), valid=valid,
                point_indices=array(output['point_indices']), winner_query=trace['winner_query'],
                pre_roi_packed=np.packbits(trace['pre'], axis=1, bitorder='little'),
                post_roi_packed=np.packbits(trace['post'], axis=1, bitorder='little'),
                final_query_indices=ids, final_class=fc,
                final_full_packed=np.packbits(full, axis=1, bitorder='little'))
            for row in trace['rows']:
                q = row['query']
                row['before'] = describe(trace['pre'][q], full_gt, gc, pan, valid)
                row['after'] = describe(trace['post'][q], full_gt, gc, pan, valid)
            write(dest/'queries.json', trace['rows'])
            comparison = cpu_iou(old, full)
            # Pair reference cases by geometry + class; query IDs are frame-local.
            matched_ref = dict(pairs(comparison, ref['final_class'], fc))
            for case in (c for c in cases if c['frame'] == f and c['arm'] == arm):
                slot = case['final_slot']
                j = slot if exact else matched_ref.get(slot)
                if j is None or comparison[slot, j] < .95:
                    reviews.append(dict(id=case['id'], matched_on_4090=False,
                                        reason='reference_geometry_not_reproduced_at_IoU_0.95'))
                    continue
                q = int(ids[j])
                r = trace['rows'][q]
                reviews.append(dict(id=case['id'], matched_on_4090=True, reference_exact=exact,
                    reference_to_trace_iou=float(comparison[slot,j]), final_slot=int(j), query=q,
                    class_probability=r['class_probability'], no_object_probability=r['no_object_probability'],
                    retention_ratio=r['retention_ratio'], before=r['before'], after=r['after'],
                    lost_to_queries=r['lost_to_queries']))
            record = dict(arm=arm, frame=f, reference_final_exact=exact,
                reference_raw_different_bits=raw_diff,
                class_probability_max_abs_difference=float(np.abs(trace['class_probability']-ref['class_probability']).max()),
                queries=len(trace['rows']), retained=len(ids), tp=len(matched), fp=len(full)-len(matched),
                reasons=dict(Counter(r['reason'] for r in trace['rows'])),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.monotonic()-start,
                trace_sha256=sha256_file(dest/'trace.npz'))
            write(dest/'summary.json', record)
            summaries.append(record)
            write(a.out/'progress.json', dict(forwards=forwards, planned_forwards=3*len(a.arms), latest=record))
            print(json.dumps(record), flush=True)
            del output, native, trace
        assert before == digest(model)
        del model
        torch.cuda.empty_cache()
    assert forwards == 3*len(a.arms)
    assert len(reviews) == sum(c['arm'] in a.arms for c in cases)
    write(a.out/'case_traces.json', reviews)
    write(a.out/'completion.json', dict(complete=True, model_forwards=forwards, optimizer_updates=0,
        weights_unchanged=True, native_extraction_equivalent=True, case_count=len(reviews),
        matched_cases=sum(r['matched_on_4090'] for r in reviews),
        all_reference_finals_exact=all(r['reference_final_exact'] for r in summaries), records=summaries))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'review', 'parent', 'weights', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--arms', nargs='+', choices=('F', 'C0', 'C1'), default=['F', 'C0', 'C1'])
    run(p.parse_args())
