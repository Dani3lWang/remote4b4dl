"""Public model-input contracts and feature interventions independent of weights."""
from pathlib import Path
import numpy as np

from . import SCHEMA
from .io import digest, read_json, read_jsonl, safe_file, sha256, unique_index
from .geometry import MOTIONS, SECTORS
from .rewards import score_response

PUBLIC_KEYS = {"schema", "request_id", "scene_id", "split", "task", "predicate", "question", "frames",
               "feature_length", "kind", "options", "evidence_keys", "response_format", "prompt",
               "input_fingerprint", "semantic_id", "option_rotation"}


def validate_request(request, manifest):
    required = PUBLIC_KEYS - {"semantic_id", "option_rotation"}
    if set(request) - PUBLIC_KEYS or not required <= set(request) or request["schema"] != SCHEMA:
        raise ValueError("unknown or supervision-only field in model request")
    task, kind = request["task"], request["kind"]
    if task not in ("existence", "position", "motion", "distance") or kind != ("mcq" if task == "distance" else "binary"):
        raise ValueError("invalid task/kind")
    if task == "motion" and request["predicate"] not in MOTIONS or task == "position" and request["predicate"] not in SECTORS:
        raise ValueError("unknown geometric predicate")
    expected = {"existence": ["count"], "position": ["x", "y"], "motion": ["d0", "d1", "delta_d", "delta_y"]}
    if task == "distance":
        if set(request["options"]) != set("ABCD") or len(set(request["options"].values())) != 4:
            raise ValueError("invalid distance options")
        keys = sorted("distance_" + x for x in request["options"].values())
        if "semantic_id" not in request or type(request.get("option_rotation")) is not int or request["option_rotation"] not in range(4):
            raise ValueError("invalid option orbit")
    else:
        keys = expected[task]
        if request["options"]:
            raise ValueError("binary request has options")
    if request["evidence_keys"] != keys or request["response_format"] != "geometry_json_v1":
        raise ValueError("invalid evidence contract")
    fingerprint = digest({k: v for k, v in request.items() if k not in ("request_id", "input_fingerprint")})
    if fingerprint != request["input_fingerprint"]:
        raise ValueError("request fingerprint mismatch")
    scene = manifest["scenes"][request["scene_id"]]
    if request["split"] != scene["split"] or request["feature_length"] != scene["frames"]:
        raise ValueError("scene/split/frame count mismatch")
    if not request["frames"] or any(type(i) is not int or not 0 <= i < scene["frames"] for i in request["frames"]):
        raise ValueError("question frame outside scene")
    if request["frames"] != sorted(set(request["frames"])):
        raise ValueError("invalid question frame ordering")


class GeometryDataset:
    def __init__(self, root, split=None):
        self.root = Path(root)
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest["schema"] != SCHEMA or self.manifest["dataset_id"] != digest({k: v for k, v in self.manifest.items() if k != "dataset_id"}):
            raise ValueError("dataset manifest identity mismatch")
        for name, expected in self.manifest["files"].items():
            if sha256(safe_file(self.root, name)) != expected:
                raise ValueError("dataset file hash mismatch: " + name)
        requests = read_jsonl(self.root / "requests.jsonl")
        unique_index(requests, "request_id")
        for request in requests:
            validate_request(request, self.manifest)
        if split not in (None, "train", "internal_dev", "test"):
            raise ValueError("unknown split")
        self.requests = [r for r in requests if split is None or r["split"] == split]
        self.targets = unique_index(read_jsonl(self.root / "supervision.jsonl"), "request_id")
        if set(self.targets) != {r["request_id"] for r in requests}:
            raise ValueError("supervision coverage mismatch")
        for request in requests:
            score_response(request, self.targets[request["request_id"]], "")
        self._cache = {}

    def scene(self, ident):
        if ident not in self._cache:
            record = self.manifest["scenes"][ident]
            path = safe_file(self.root, record["bundle"])
            if sha256(path) != record["sha256"]:
                raise ValueError("scene bundle hash mismatch")
            with np.load(path, allow_pickle=False) as data:
                bundle = {key: data[key] for key in data.files}
            n = record["frames"]
            if set(bundle) != {"features", "frame_times", "frame_geometry", "ego_from_lidar"} or bundle["features"].shape != (n, 768):
                raise ValueError("invalid scene bundle")
            if bundle["frame_times"].shape != (n,) or bundle["frame_geometry"].shape != (n, 4, 4) or bundle["ego_from_lidar"].shape != (n, 4, 4):
                raise ValueError("invalid geometry axes")
            if not all(np.isfinite(x).all() for x in bundle.values()) or np.any(np.diff(bundle["frame_times"]) <= 0):
                raise ValueError("invalid frame values")
            for key in ("frame_geometry", "ego_from_lidar"):
                p = bundle[key]
                r = p[:, :3, :3]
                if not np.allclose(p[:, 3], [0, 0, 0, 1], atol=1e-5) or not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(r), 1, atol=1e-4):
                    raise ValueError("invalid rigid pose bundle")
            stamps = np.asarray(record["timestamps_us"], dtype=np.int64)
            if not np.allclose(bundle["frame_times"], (stamps-stamps[0])/1e6, atol=1e-9, rtol=0):
                raise ValueError("timestamp mapping mismatch")
            self._cache[ident] = bundle
        return {key: value.copy() for key, value in self._cache[ident].items()}

    def model_input(self, request, control=None):
        validate_request(request, self.manifest)
        scene = self.scene(request["scene_id"])
        if control is not None:
            if control["dataset_id"] != self.manifest["dataset_id"] or control["request_id"] != request["request_id"] or control["input_fingerprint"] != request["input_fingerprint"]:
                raise ValueError("control/request mismatch")
            if control["status"] != "ready":
                raise ValueError("control unavailable: " + control["status"])
            donor = None
            if control["condition"] == "cross_scene":
                ident = control["donor_scene_id"]
                record = self.manifest["scenes"][ident]
                if ident == request["scene_id"] or record["split"] != request["split"] or record["frames"] != request["feature_length"]:
                    raise ValueError("donor must be another same-split, equal-length scene")
                donor = self.scene(ident)["features"]
            scene["features"] = intervene(scene["features"], control["condition"], control.get("permutation"), donor)
        return {"prompt": request["prompt"], "features": scene["features"],
                "frame_times": scene["frame_times"], "frame_geometry": scene["frame_geometry"],
                "frame_valid_mask": np.ones(len(scene["features"]), dtype=bool)}


def intervene(features, condition, permutation=None, donor=None):
    if condition == "identity":
        return features.copy()
    if condition == "zero":
        return np.zeros_like(features)
    if condition == "repeat_first":
        return np.repeat(features[:1], len(features), axis=0)
    if condition == "shuffle_features":
        if permutation is None or any(type(i) is not int for i in permutation) or sorted(permutation) != list(range(len(features))) or permutation == list(range(len(features))):
            raise ValueError("non-identity full feature permutation required")
        return features[permutation].copy()
    if condition == "cross_scene":
        if donor is None or donor.shape != features.shape or not np.isfinite(donor).all():
            raise ValueError("matched donor features required")
        return donor.astype(features.dtype, copy=True)
    raise ValueError("unknown intervention")


def control_plan(dataset, condition, seed=42):
    if condition not in ("identity", "zero", "repeat_first", "shuffle_features", "cross_scene"):
        raise ValueError("unknown condition")
    result = []
    for request in dataset.requests:
        row = {"request_id": request["request_id"], "input_fingerprint": request["input_fingerprint"],
               "dataset_id": dataset.manifest["dataset_id"], "condition": condition, "status": "ready", "seed": seed}
        ident, n = request["scene_id"], request["feature_length"]
        if condition == "shuffle_features":
            rng = np.random.default_rng(int(digest([seed, ident])[:16], 16))
            order = rng.permutation(n).tolist()
            if order == list(range(n)):
                order = order[1:] + order[:1]
            row["permutation"] = order
            if n < 2:
                row["status"] = "unavailable_single_frame"
        if condition == "cross_scene":
            candidates = sorted(s for s, rec in dataset.manifest["scenes"].items()
                                if s != ident and rec["split"] == request["split"] and rec["frames"] == n)
            if candidates:
                row["donor_scene_id"] = candidates[int(digest([seed, ident])[:16], 16) % len(candidates)]
            else:
                row["status"] = "unavailable_equal_length_donor"
        result.append(row)
    return result


def collate_frames(items):
    import torch
    if not items:
        raise ValueError("empty batch")
    n, width = max(len(x["features"]) for x in items), items[0]["features"].shape[1]
    features = torch.zeros(len(items), n, width)
    times = torch.zeros(len(items), n, dtype=torch.float64)
    poses = torch.eye(4, dtype=torch.float64).repeat(len(items), n, 1, 1)
    valid = torch.zeros(len(items), n, dtype=torch.bool)
    for i, item in enumerate(items):
        size = len(item["features"])
        features[i, :size] = torch.as_tensor(item["features"])
        times[i, :size] = torch.as_tensor(item["frame_times"])
        poses[i, :size] = torch.as_tensor(item["frame_geometry"])
        valid[i, :size] = True
    return {"features": features, "frame_times": times, "frame_geometry": poses, "frame_valid_mask": valid}
