"""B4DL-facing object candidates; query IDs are frame-local, never trajectory IDs."""
import json
from pathlib import Path

import numpy as np
import torch

from grounded4d.taxonomy import THING_CLASSES


@torch.no_grad()
def object_records(frame, instances):
    """Use predictions only; labels and GT instance tokens are never read here."""
    rows = []
    for position, query in enumerate(instances["query_indices"].tolist()):
        mask = instances["masks"][position]
        center = frame["points"][mask, :3].mean(0).cpu().tolist()
        rows.append(dict(object_id=f'{frame["sample_data_token"]}:q{query}',
                         query_index=query, feature_row=position,
                         class_id=int(instances["class_ids"][position]),
                         class_name=THING_CLASSES[int(instances["class_ids"][position])],
                         score=float(instances["scores"][position]),
                         point_count=int(mask.sum()), center_lidar=center,
                         track_id=None))
    return rows


def resolve_answer_objects(records, selected_ids):
    """Resolve externally generated object IDs without category/nearest-instance guessing."""
    lookup = {row["object_id"]: row for row in records}
    if len(lookup) != len(records) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("duplicate object IDs")
    if any(item not in lookup for item in selected_ids):
        raise ValueError("answer refers to an unknown or different-frame object ID")
    return [lookup[item] for item in selected_ids]


@torch.no_grad()
def save_prediction(directory, frame, output, instances, identity, save_all_queries=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    token = frame["sample_data_token"]
    if not token.isalnum():
        raise ValueError("sample_data_token must be a safe frame identifier")
    path = directory / (token + ".npz")
    record_path = directory / (token + ".json")
    if path.exists() or record_path.exists():
        raise FileExistsError("prediction already exists: " + token)
    arrays = dict(point_indices=np.arange(len(frame["points"]), dtype=np.int64),
                  valid_points=output["valid_points"].cpu().numpy(),
                  masks=instances["masks"].cpu().numpy(),
                  class_ids=instances["class_ids"].cpu().numpy(),
                  scores=instances["scores"].cpu().numpy(),
                  query_indices=instances["query_indices"].cpu().numpy(),
                  query_features=instances["query_features"].cpu().numpy(),
                  point_sha256=np.asarray(frame["point_sha256"]))
    if save_all_queries:
        arrays.update(all_mask_probabilities=output["mask_logits"].float().sigmoid().cpu().numpy(),
                      all_class_probabilities=instances["all_class_probabilities"].cpu().numpy(),
                      all_query_features=output["queries"].cpu().numpy())
    temporary = path.with_suffix(".npz.part")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    record = dict(schema_version="b4dl-maskpls-objects-v1",
                  sample_data_token=token, sample_token=frame["sample_token"],
                  scene_token=frame.get("scene_token"), timestamp_us=frame.get("timestamp_us"),
                  point_sha256=frame["point_sha256"], mask_file=path.name,
                  coordinate_frame="native_lidar", point_order="original_nuscenes_rows",
                  objects=object_records(frame, instances), identity=identity,
                  grounding_verified=False, track_verified=False)
    temporary_json = record_path.with_suffix(".json.part")
    temporary_json.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")
    temporary_json.replace(record_path)
