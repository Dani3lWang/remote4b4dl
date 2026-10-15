"""PyTorch-only orchestration; predictions never consume masks or reference targets."""
import hashlib
from pathlib import Path

import torch
from torch import nn

from .config import NO_OBJECT, UPSTREAM_COMMIT
from .decoder import MaskedTransformerDecoder


class MaskPLS(nn.Module):
    def __init__(self, config, *, backbone=None):
        super().__init__()
        config.validate()
        self.config = config
        bb, dec, data = config.upstream()
        if backbone is None:
            # Optional native extension: no dependency on MinkowskiEngine for data audits.
            from .backbone import MinkEncoderDecoder
            backbone = MinkEncoderDecoder(bb, data)
        self.backbone = backbone
        self.decoder = MaskedTransformerDecoder(dec, bb, data)

    def forward(self, points):
        if points.ndim != 2 or points.shape[1] != 4 or not torch.isfinite(points).all():
            raise ValueError("expected finite, original-order [N,4] XYZ/intensity")
        if points.dtype != torch.float32:
            raise ValueError("sparse spatial path requires FP32")
        lower = points.new_tensor(self.config.point_cloud_range[:3])
        upper = points.new_tensor(self.config.point_cloud_range[3:])
        valid = ((points[:, :3] >= lower) & (points[:, :3] < upper)).all(1)
        indices = valid.nonzero(as_tuple=True)[0]
        if not len(indices):
            raise ValueError("no points inside configured ROI")
        values = points.clone()
        if self.config.intensity_mode == "frame_minmax":
            # Converter normalizes the FULL frame before the model applies its ROI.
            intensity = values[:, 3]
            span = intensity.max() - intensity.min()
            values[:, 3] = (intensity - intensity.min()) / span.clamp_min(1e-8)
        feats, coors, padding, semantic = self.backbone(values[indices])
        outputs, pad = self.decoder(feats, coors, padding)
        if pad.shape != (1, len(indices)) or pad.any():
            raise ValueError("single-frame backbone lost input point count/order")

        def restore(item):
            logits = item["pred_masks"][0].T
            full = logits.new_full((len(logits), len(points)), -1e4)
            full[:, indices] = logits
            return dict(mask_logits=full, class_logits=item["pred_logits"][0])

        result = restore(outputs)
        sem_full = semantic.new_zeros((len(points), semantic.shape[-1]))
        sem_full[indices] = semantic[0]
        return dict(**result, queries=outputs["queries"][0], semantic_logits=sem_full,
                    valid_points=valid, point_indices=indices,
                    auxiliary=[restore(a) for a in outputs["aux_outputs"]])


def import_official_state(model, state, *, allow_query_expansion=False):
    """Exact backbone/decoder loading; only explicit query expansion is permitted."""
    wanted = model.state_dict()
    selected, ignored = {}, []
    for key, value in state.items():
        if key.startswith(("mask_loss.", "sem_loss.")):
            ignored.append(key)  # loss buffers, not model parameters
        elif key in wanted:
            selected[key] = value
        else:
            raise ValueError("unexpected official checkpoint key: " + key)
    missing = set(wanted) - set(selected)
    if missing:
        raise ValueError("missing official weights: " + ", ".join(sorted(missing)))
    expanded = {}
    for key, target in wanted.items():
        source = selected[key]
        if source.shape == target.shape:
            continue
        if (allow_query_expansion
                and key in ("decoder.query_feat.weight", "decoder.query_embed.weight")
                and source.ndim == 2 and source.shape[1] == target.shape[1]
                and source.shape[0] < target.shape[0]):
            merged = target.clone()
            merged[:len(source)] = source
            selected[key] = merged
            expanded[key] = dict(pretrained_rows=len(source), new_rows=len(target) - len(source))
        else:
            raise ValueError(f"official weight shape mismatch: {key}: {source.shape} != {target.shape}")
    model.load_state_dict(selected, strict=True)
    return dict(upstream_commit=UPSTREAM_COMMIT, query_expansion=expanded,
                ignored_loss_buffers=ignored, preprocessing_equivalence_verified=False)


def load_official(model, path, *, allow_query_expansion=False):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    state = raw.get("state_dict", raw)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("expected official state_dict or a tensor-only weights export")
    result = import_official_state(model, state, allow_query_expansion=allow_query_expansion)
    result["checkpoint_sha256"] = sha256_file(path)
    result["evaluation_scope"] = "warm_start_or_diagnostic; official train scenes may have been seen"
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def extract_instances(output, config):
    """Panoptic competition includes stuff; B4DL exports only ten thing classes."""
    if not all(torch.isfinite(output[key]).all() for key in ("class_logits", "mask_logits", "queries")):
        raise FloatingPointError("nonfinite predictions cannot be reported as empty/NOOBJ")
    probabilities = output["class_logits"].float().softmax(-1)
    scores, classes = probabilities.max(-1)
    mask_probability = output["mask_logits"].float().sigmoid()
    valid = output["valid_points"]
    eligible = (classes > 0) & (classes < NO_OBJECT) & (scores >= config.score_threshold)
    query_ids = eligible.nonzero(as_tuple=True)[0]
    retained, masks = [], []
    if len(query_ids):
        selected = mask_probability[query_ids]
        winner = (selected * scores[query_ids, None]).argmax(0)
        for local_id, query_id in enumerate(query_ids.tolist()):
            if int(classes[query_id]) > 10:
                continue
            original = selected[local_id].ge(config.mask_threshold) & valid
            mask = original & winner.eq(local_id)
            # No 10/50 point exclusion: even a single surviving point is exported.
            if mask.any() and mask.sum() / original.sum().clamp_min(1) >= config.overlap_threshold:
                retained.append(query_id)
                masks.append(mask)
    ids = torch.tensor(retained, dtype=torch.long, device=scores.device)
    binary = torch.stack(masks) if masks else torch.empty((0, len(valid)), dtype=torch.bool, device=valid.device)
    return dict(query_indices=ids, masks=binary, class_ids=classes[ids] - 1,
                scores=scores[ids], query_features=output["queries"][ids],
                all_class_probabilities=probabilities, all_classes=classes)
