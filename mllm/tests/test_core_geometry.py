import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from b4dl_core.data import (assign_splits, balance_questions, build_dataset, generate_scene_questions,
                           load_aligned_scenes, make_question, nearest)
from b4dl_core.audit import audit_supervision
from b4dl_core.evaluation import compare_reports, evaluate, opposite_pairs, prior_predictions, scene_interval
from b4dl_core.geometry import global_to_ego, motion_truth, motion_values, sector, transform
from b4dl_core.inputs import GeometryDataset, collate_frames, control_plan, intervene, validate_request
from b4dl_core.io import digest, iter_array, read_json, write_json
from b4dl_core.rewards import parse_response, score_response
from b4dl_core.temporal import FrameTemporalAdapter, TemporalConfig, TemporalProjector, relative_geometry


def obj(ident, x, y=0, category="car"):
    return {"instance_token": ident, "annotation_token": ident + "-ann", "class": category,
            "xyz": [x, y, 0], "distance": math.hypot(x, y), "num_lidar_pts": 5}


def scene(ident="s0", split="test", n=10):
    return {"scene_id": ident, "split": split, "frames": [
        {"index": i, "timestamp_us": i * 500000, "objects": [obj("a", 10-i*.5)], "unsupported_classes": []}
        for i in range(n)]}


def binary(ident="s0", answer="Yes", split="test"):
    return make_question(scene(ident, split), "existence", "Is there a car?", answer,
                         {"count": 1 if answer == "Yes" else 0}, [0])


def fake_dataset(pairs):
    return SimpleNamespace(requests=[r for r, _ in pairs], targets={r["request_id"]: t for r, t in pairs},
                           manifest={"dataset_id": "fixture"})


def prediction(request, target, **changes):
    result = {"request_id": request["request_id"], "input_fingerprint": request["input_fingerprint"],
              "status": "ok", "text": json.dumps({k: target[k] for k in ("answer", "evidence")})}
    result.update(changes)
    return result


def run_manifest():
    return {"dataset_id": "fixture", "split": "test", "condition": "identity", "model_id": "unit-fixture",
            "checkpoint_sha256": None, "prediction_kind": "fixture"}


class GeometryTests(unittest.TestCase):
    def test_coordinate_transform_and_relative_not_world_motion(self):
        p0 = {"translation": [100, 200, 0], "rotation": [1, 0, 0, 0]}
        p1 = {"translation": [103, 200, 0], "rotation": [1, 0, 0, 0]}
        start, end = (global_to_ego([110, 200, 0], p) for p in (p0, p1))
        self.assertEqual(motion_values(start, end)["delta_d"], -3)
        self.assertTrue(motion_truth(motion_values(start, end))["closer"])
        p1["rotation"] = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
        np.testing.assert_allclose(global_to_ego([103, 205, 0], p1), [5, 0, 0], atol=1e-12)

    def test_thresholds_and_sector_boundaries(self):
        self.assertTrue(motion_truth({"delta_d": -2, "delta_y": 1})["closer"])
        self.assertTrue(motion_truth({"delta_d": 2, "delta_y": -1})["right"])
        self.assertFalse(motion_truth({"delta_d": 1, "delta_y": 0})["similar"])
        self.assertFalse(motion_truth({"delta_d": 0, "delta_y": .5})["similar"])
        self.assertEqual(sector(1, 0), "front")
        self.assertEqual(sector(0, 1), "left")
        self.assertEqual(sector(-1, 0), "back")
        self.assertEqual(sector(1, math.tan(math.radians(22.50001))), "front-left")
        with self.assertRaises(ValueError):
            sector(0, 0)

    def test_follow_same_instance_despite_nearest_switch(self):
        s = scene()
        s["frames"][8]["objects"] = [obj("a", 15), obj("b", 1)]
        rows = generate_scene_questions(s)
        values = [(r, t) for r, t in rows if r["task"] == "motion" and r["frames"] == [0, 8]]
        self.assertEqual(len(values), 5)
        self.assertTrue(all(t["evidence"]["d1"] == 15 and t["provenance"]["instances"] == ["a"] for _, t in values))
        s["frames"][8]["objects"] = [obj("b", 1)]
        self.assertFalse(any(r["task"] == "motion" and r["frames"] == [0, 8] for r, _ in generate_scene_questions(s)))

    def test_zero_returns_unknown_and_nearest_ties(self):
        s = scene(n=1)
        s["frames"][0]["objects"] = []
        s["frames"][0]["unsupported_classes"] = ["car"]
        rows = generate_scene_questions(s)
        self.assertFalse(any(" car " in r["question"] for r, _ in rows))
        self.assertEqual(nearest([obj("a", 1), obj("b", -1)]), {})

    def test_scene_splits_stable_for_subsets(self):
        rows = [{"token": str(i), "name": "scene-" + str(i)} for i in range(6)]
        official = {"version": "test", "source": "fixture", "train": [r["name"] for r in rows[:5]], "validation": [rows[5]["name"]]}
        full = assign_splits(rows, official)
        self.assertEqual(assign_splits(rows[1:3], official), {r["token"]: full[r["token"]] for r in rows[1:3]})
        self.assertEqual(full["5"], "test")
        official["validation"].append("scene-0")
        with self.assertRaises(ValueError):
            assign_splits(rows, official)

    def test_balance_and_option_rotations(self):
        s = scene(n=1)
        s["frames"][0]["objects"] = [obj("a", 1, category="car"), obj("b", 2, category="bus"),
                                         obj("c", 3, category="truck"), obj("d", 4, category="bicycle")]
        rows, shortages = balance_questions(generate_scene_questions(s))
        distance = [(r, t) for r, t in rows if r["task"] == "distance"]
        self.assertEqual({t["answer"] for _, t in distance}, set("ABCD"))
        self.assertEqual(len({r["semantic_id"] for r, _ in distance}), 1)
        self.assertTrue(all(r["options"][t["answer"]] == "car" for r, t in distance))


class TemporalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        self.cfg = TemporalConfig(width=16, heads=4, layers=2, ff_width=32)
        self.x = torch.randn(2, 5, 16)
        self.times = torch.tensor([[0., .5, 1., 1.5, 2.], [0., .5, 1., 0., 0.]], dtype=torch.float64)
        self.poses = torch.eye(4, dtype=torch.float64).repeat(2, 5, 1, 1)
        self.poses[0, :, 0, 3] = torch.arange(5)
        self.valid = torch.tensor([[1]*5, [1, 1, 1, 0, 0]], dtype=torch.bool)

    def call(self, model):
        return model(self.x, self.times, self.poses, self.valid)

    def test_zero_init_and_disabled_projector(self):
        m = FrameTemporalAdapter(self.cfg)
        self.assertTrue(torch.equal(self.call(m)[self.valid], self.x[self.valid]))
        self.assertTrue(torch.equal(self.call(m)[~self.valid], torch.zeros_like(self.x[~self.valid])))
        p = torch.nn.Linear(16, 7)
        self.assertTrue(torch.equal(TemporalProjector(p)(self.x), p(self.x)))
        with self.assertRaises(ValueError):
            TemporalProjector(p, m)(self.x)

    def test_padding_nan_and_single_equivalence_after_activation(self):
        m = FrameTemporalAdapter(self.cfg)
        torch.nn.init.normal_(m.output.weight, std=.1)
        self.x[1, 3:] = float("nan")
        self.poses[1, 3:] = float("nan")
        self.times[1, 3:] = float("nan")
        batch = self.call(m)
        single = m(self.x[1:2, :3], self.times[1:2, :3], self.poses[1:2, :3])
        torch.testing.assert_close(batch[1, :3], single[0], atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.isfinite(batch).all())

    def test_pair_formula_global_rigid_invariance_and_geometry_effect(self):
        cfg = self.cfg
        relation = relative_geometry(self.times, self.poses, self.valid, cfg)
        self.assertAlmostEqual(float(relation[0, 0, 1, 0]), .1)
        g = torch.tensor(transform({"translation": [100, 200, -5], "rotation": [math.sqrt(.5), 0, 0, math.sqrt(.5)]}))
        moved = g @ self.poses
        torch.testing.assert_close(relation[self.valid], relative_geometry(self.times + 123456789., moved, self.valid, cfg)[self.valid], atol=1e-6, rtol=1e-6)
        m = FrameTemporalAdapter(cfg)
        torch.nn.init.normal_(m.output.weight, std=.1)
        modified = self.poses.clone()
        modified[0, 2, 1, 3] = 5
        self.assertFalse(torch.allclose(self.call(m), m(self.x, self.times, modified, self.valid)))
        torch.testing.assert_close(self.call(m)[self.valid], m(self.x, self.times, moved, self.valid)[self.valid], atol=1e-6, rtol=1e-5)

    def test_gradient_after_residual_warmup_and_roundtrip(self):
        m = FrameTemporalAdapter(self.cfg)
        opt = torch.optim.SGD(m.parameters(), lr=.05)
        self.call(m)[self.valid].square().mean().backward()
        self.assertEqual(float(m.blocks[0].attention.in_proj_weight.grad.abs().sum()), 0.)
        self.assertGreater(float(m.output.weight.grad.abs().sum()), 0.)
        opt.step()
        opt.zero_grad()
        self.call(m)[self.valid].square().mean().backward()
        self.assertGreater(float(m.blocks[0].attention.in_proj_weight.grad.abs().sum()), 0.)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"adapter.pt"
            m.save(path)
            loaded = FrameTemporalAdapter.load(path, self.cfg)
            torch.testing.assert_close(self.call(m), self.call(loaded), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                FrameTemporalAdapter.load(path, TemporalConfig(width=16, heads=4, layers=1))

    def test_fail_closed_bad_times_mask_pose_and_width(self):
        m = FrameTemporalAdapter(self.cfg)
        for field in ("times", "valid", "poses", "x"):
            original = getattr(self, field).clone()
            if field == "times": self.times[0, 2] = self.times[0, 1]
            if field == "valid": self.valid[0, 1] = False
            if field == "poses": self.poses[0, 0, 0, 0] = 2
            if field == "x": self.x[0, 0, 0] = float("inf")
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.call(m)
            setattr(self, field, original)
        with self.assertRaises(ValueError):
            m(self.x[:, :, :8], self.times, self.poses, self.valid)

    def test_time_only_configuration(self):
        m = FrameTemporalAdapter(TemporalConfig(width=16, heads=4, use_geometry=False))
        torch.nn.init.normal_(m.output.weight, std=.1)
        y = m(self.x, self.times, frame_valid_mask=self.valid)
        self.assertTrue(torch.isfinite(y).all())
        torch.testing.assert_close(y, self.call(m))


class RewardAndEvaluationTests(unittest.TestCase):
    def test_strict_response_parser(self):
        r, t = binary()
        good = prediction(r, t)["text"]
        self.assertTrue(parse_response(good, r)["evidence_valid"])
        for value in ('prefix ' + good, '{"answer":"Yes","answer":"No","evidence":{"count":1}}',
                      '{"answer":"Yes","evidence":{"count":NaN}}', '{"answer":"Yes","evidence":{"count":true}}',
                      '{"answer":"Yes","evidence":{"count":1.0}}', '{"answer":"Yes","evidence":{"count":-1}}',
                      '{"answer":"Yes","evidence":{"count":1,"x":2}}', '[]'):
            with self.subTest(value=value):
                self.assertFalse(parse_response(value, r)["evidence_valid"])

    def test_fabricated_and_contradictory_evidence(self):
        r, t = make_question(scene(), "position", "front?", "Yes", {"x": 10, "y": 0}, [0], predicate="front")
        perfect = score_response(r, t, prediction(r, t)["text"])
        self.assertEqual(perfect["combined_reward"], 1.)
        fake = score_response(r, t, json.dumps({"answer": "Yes", "evidence": {"x": 100, "y": 20}}))
        self.assertTrue(fake["evidence_consistent"])
        self.assertEqual(fake["evidence_score"], 0.)
        conflict = score_response(r, t, json.dumps({"answer": "No", "evidence": t["evidence"]}))
        self.assertEqual(conflict["combined_reward"], 0.)
        malformed = copy.deepcopy(t)
        malformed["evidence"]["x"] = True
        with self.assertRaises(ValueError):
            score_response(r, malformed, "")

    def test_motion_arithmetic_invalid(self):
        r, t = make_question(scene(), "motion", "closer?", "Yes", {"d0": 10, "d1": 5, "delta_d": -5, "delta_y": 0}, [0, 8], predicate="closer")
        value = {"answer": "Yes", "evidence": {**t["evidence"], "d1": 9}}
        self.assertEqual(score_response(r, t, json.dumps(value))["evidence_score"], 0)

    def test_full_denominator_and_prediction_identity(self):
        pairs = [binary("s0"), binary("s1", "No"), binary("s2")]
        d = fake_dataset(pairs)
        good = prediction(*pairs[0])
        error = prediction(*pairs[1], status="error")
        report = evaluate(d, [good, error], run_manifest(), repetitions=30)
        self.assertEqual(report["summary"]["n"], 3)
        self.assertEqual(report["summary"]["accuracy"], 1/3)
        self.assertEqual(report["summary"]["status_counts"], {"ok": 1, "error": 1, "missing": 1})
        for predictions in ([good, good], [{**good, "request_id": "unknown"}], [{**good, "input_fingerprint": "wrong"}]):
            with self.assertRaises(ValueError):
                evaluate(d, predictions, run_manifest())
        run = {**run_manifest(), "prediction_kind": "model"}
        with self.assertRaises(ValueError):
            evaluate(d, [good], run)

    def test_pairing_maximum_disjoint_and_scene_constraints(self):
        pairs = [binary("s0", "Yes"), binary("s1", "Yes"), binary("s1", "No"), binary("s2", "No")]
        # Make IDs unique when the same scene has opposite synthetic labels.
        pairs[2][0]["request_id"] += "-no"
        pairs[2][1]["request_id"] = pairs[2][0]["request_id"]
        d = fake_dataset(pairs)
        result = opposite_pairs(d.requests, d.targets)
        self.assertEqual(len(result), 2)
        ids = [p[k] for p in result for k in ("yes_id", "no_id")]
        self.assertEqual(len(set(ids)), 4)
        index = {r["request_id"]: r for r in d.requests}
        self.assertTrue(all(index[p["yes_id"]]["scene_id"] != index[p["no_id"]]["scene_id"] for p in result))
        self.assertEqual(result, opposite_pairs(list(reversed(d.requests)), d.targets))
        d.requests[0]["prompt"] += "different"
        self.assertEqual(len(opposite_pairs(d.requests, d.targets)), 1)

    def test_option_rotation_metric_semantic_not_letter(self):
        s = scene(n=1)
        s["frames"][0]["objects"] = [obj(str(i), i+1, category=c) for i, c in enumerate(("car", "bus", "truck", "bicycle"))]
        pairs = [(r, t) for r, t in generate_scene_questions(s) if r["task"] == "distance"]
        d = fake_dataset(pairs)
        good = evaluate(d, [prediction(r, t) for r, t in pairs], run_manifest(), repetitions=20)
        fixed = [prediction(r, {**t, "answer": "A"}) for r, t in pairs]
        bad = evaluate(d, fixed, run_manifest(), repetitions=20)
        self.assertEqual(good["option_rotations"]["semantic_consistency"], 1)
        self.assertEqual(bad["option_rotations"]["semantic_consistency"], 0)
        self.assertEqual(bad["summary"]["accuracy"], .25)

    def test_paired_delta_bootstrap_and_priors(self):
        pairs = [binary("s0"), binary("s1", "No")]
        d = fake_dataset(pairs)
        first = evaluate(d, [prediction(*pairs[0])], run_manifest(), repetitions=20)
        second = evaluate(d, [prediction(*pairs[1])], run_manifest(), repetitions=20)
        result = compare_reports(first, second, 30)
        self.assertEqual((result["repairs"], result["regressions"], result["accuracy_delta"]), (1, 1, 0))
        self.assertEqual(scene_interval(first["rows"], repetitions=30), scene_interval(first["rows"], repetitions=30))
        self.assertIsNone(scene_interval(first["rows"][:1])["low"])
        with self.assertRaises(ValueError):
            prior_predictions(d, d)
        train = fake_dataset([binary("train0", "No", "train")])
        predictions = prior_predictions(train, d)
        self.assertTrue(all(json.loads(p["text"])["answer"] == "No" for p in predictions))


class InputTests(unittest.TestCase):
    def test_interventions_really_change_features(self):
        x = np.arange(12).reshape(3, 4).astype(np.float32)
        np.testing.assert_array_equal(intervene(x, "identity"), x)
        for cond, kwargs in (("zero", {}), ("repeat_first", {}), ("shuffle_features", {"permutation": [2, 0, 1]}),
                             ("cross_scene", {"donor": x+100})):
            y = intervene(x, cond, **kwargs)
            self.assertFalse(np.array_equal(x, y))
            self.assertFalse(np.shares_memory(x, y))
        for perm in ([0, 1, 2], [0, 0, 1], [0, 1], [False, 1, 2]):
            with self.assertRaises(ValueError):
                intervene(x, "shuffle_features", perm)

    def test_control_plan_same_split_and_deterministic(self):
        d = fake_dataset([binary()])
        d.manifest["scenes"] = {"s0": {"split": "test", "frames": 10}, "s1": {"split": "train", "frames": 10},
                                "s2": {"split": "test", "frames": 9}}
        self.assertEqual(control_plan(d, "cross_scene")[0]["status"], "unavailable_equal_length_donor")
        d.manifest["scenes"]["s2"]["frames"] = 10
        self.assertEqual(control_plan(d, "cross_scene")[0]["donor_scene_id"], "s2")
        self.assertEqual(control_plan(d, "shuffle_features"), control_plan(d, "shuffle_features"))

    def test_no_gt_fields_and_fingerprint_tampering(self):
        r, _ = binary()
        manifest = {"scenes": {"s0": {"split": "test", "frames": 10}}}
        validate_request(r, manifest)
        for key, value in (("answer", "Yes"), ("question", "edited"), ("frames", [100])):
            with self.assertRaises(ValueError):
                validate_request({**r, key: value}, manifest)

    def test_stream_array_chunk_boundary_and_malformed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"rows.json"
            values = [{"text": "x"*70000}, {"中文": 1}]
            write_json(path, values)
            self.assertEqual(list(iter_array(path)), values)
            for text in ('[{"a":1},]', '[{"a":1}', '[{"a":1}] trailing', '{}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    list(iter_array(path))


def raw_fixture(root):
    tables, features = root/"tables", root/"features"
    tables.mkdir(); features.mkdir()
    metadata = root/"metadata.json"
    official = root/"splits.json"
    write_json(official, {"version": "fixture", "source": "unit-test", "train": ["scene-0", "scene-1"], "validation": ["scene-2"]})
    rows = {k: [] for k in ("scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor", "sample_annotation", "instance", "category")}
    rows["sensor"] = [{"token": "sensor", "channel": "LIDAR_TOP"}]
    rows["calibrated_sensor"] = [{"token": "cal", "sensor_token": "sensor", "rotation": [1, 0, 0, 0], "translation": [0, 0, 1]}]
    rows["category"] = [{"token": "car", "name": "vehicle.car"}]
    metas = []
    for s in range(3):
        sid = "s" + str(s)
        rows["scene"].append({"token": sid, "name": "scene-"+str(s), "nbr_samples": 10,
                              "first_sample_token": sid+"-0", "last_sample_token": sid+"-9"})
        rows["instance"].append({"token": sid+"-obj", "category_token": "car"})
        paths = {}
        for i in range(10):
            token = f"{sid}-{i}"
            stamp = 1500000000000000 + i*500000
            rows["sample"].append({"token": token, "scene_token": sid, "prev": f"{sid}-{i-1}" if i else "",
                                   "next": f"{sid}-{i+1}" if i<9 else ""})
            rows["sample_data"].append({"token": token+"-sd", "sample_token": token, "filename": token+".bin",
                                       "timestamp": stamp, "is_key_frame": True, "ego_pose_token": token+"-pose", "calibrated_sensor_token": "cal"})
            rows["ego_pose"].append({"token": token+"-pose", "timestamp": stamp, "rotation": [1, 0, 0, 0], "translation": [i*.5, 0, 0]})
            rows["sample_annotation"].append({"token": token+"-ann", "sample_token": token,
                                             "instance_token": sid+"-obj", "translation": [10, 0, 0], "num_lidar_pts": 5})
            paths[f"PATH_{i:03d}"] = token+".bin"
        np.save(features/(sid+".npy"), np.arange(7680, dtype=np.float32).reshape(10, 768)/1000+s)
        metas.append({"scene_id": sid, "scene_token": sid, "num_frames": 10, "paths": {"PATH_LIDAR_TOP": paths}})
    for key, value in rows.items(): write_json(tables/(key+".json"), value)
    write_json(metadata, metas)
    return tables, metadata, features, official


class AlignmentTests(unittest.TestCase):
    def test_raw_alignment_build_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tables, metadata, features, official = raw_fixture(root)
            scenes = load_aligned_scenes(tables, metadata, features, read_json(official))
            self.assertEqual(scenes[0]["frames"][8]["objects"][0]["xyz"], [6, 0, 0])
            output = root/"bundle"
            manifest = build_dataset(tables, metadata, features, official, output)
            dataset = GeometryDataset(output)
            self.assertEqual(audit_supervision(dataset, tables)["questions"], len(dataset.requests))
            self.assertEqual(len(dataset.requests), manifest["selected_questions"])
            item = dataset.model_input(dataset.requests[0])
            self.assertEqual(set(item), {"prompt", "features", "frame_times", "frame_geometry", "frame_valid_mask"})
            self.assertEqual(collate_frames([item])["features"].shape, (1, 10, 768))
            data = dataset.scene("s0")
            data["features"][:] = 100
            self.assertFalse(np.all(dataset.scene("s0")["features"] == 100))
            path = output/"requests.jsonl"
            path.write_text(path.read_text()+"\n")
            with self.assertRaises(ValueError): GeometryDataset(output)

    def test_raw_alignment_rejects_bad_row_pose_time_duplicates(self):
        for kind in ("paths", "features", "pose_time", "sample_order", "duplicate_lidar", "duplicate_instance"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                tables, metadata, features, official = raw_fixture(root)
                if kind == "paths":
                    m = read_json(metadata); m[0]["paths"]["PATH_LIDAR_TOP"]["PATH_000"] = "s1-0.bin"; write_json(metadata, m)
                if kind == "features": np.save(features/"s0.npy", np.zeros((9, 768)))
                if kind == "pose_time":
                    p = read_json(tables/"ego_pose.json"); p[0]["timestamp"] += 1; write_json(tables/"ego_pose.json", p)
                if kind == "sample_order":
                    p = read_json(tables/"sample.json"); p[0]["next"] = "s0-2"; write_json(tables/"sample.json", p)
                if kind == "duplicate_lidar":
                    p = read_json(tables/"sample_data.json"); p.append({**p[0], "token": "dupe", "filename": "dupe.bin"}); write_json(tables/"sample_data.json", p)
                if kind == "duplicate_instance":
                    p = read_json(tables/"sample_annotation.json"); p.append({**p[0], "token": "dupe"}); write_json(tables/"sample_annotation.json", p)
                with self.assertRaises(ValueError):
                    load_aligned_scenes(tables, metadata, features, read_json(official))


if __name__ == "__main__":
    unittest.main()
