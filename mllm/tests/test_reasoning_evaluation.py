import ast
import copy
import math
from pathlib import Path
import re
import unittest
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

from b4dl_core.reasoning.evaluation import compare_reports, evaluate
from b4dl_core.reasoning.protocol import encode_trace
from b4dl_core.reasoning.six_query import build_free_query
from scripts.ego_text import build_metatoken_line, render_metatoken_for_range
from test_reasoning_protocol import example


class ReasoningEvaluationTests(unittest.TestCase):
    @staticmethod
    def comparison_run():
        return {"schema": "b4dl-reasoning-inference-v1", "dataset_id": "fixture", "split": "train",
                "condition": "identity", "seed": 42, "max_new_tokens": 512, "decoding": "greedy",
                "reward_version": "fixture", "runtime": {"torch": "fixture"}, "device": "cpu",
                "dtype": "float32", "gpu_name": None, "checkpoint_sha256": "same",
                "lineage": {"common": {"baseline": "same"}, "sft_origin_sha256": "same"},
                "training": {"stage": "sft", "recipe": {}, "step": 1, "execution": {"device": "cpu"},
                             "planned_full_budget": True}}

    def test_missing_and_malformed_answers_remain_in_denominator(self):
        request, target = example("existence")
        second = {**request, "request_id": "second", "scene_id": "other", "input_fingerprint": "other"}
        second_target = {**target, "request_id": "second"}
        dataset = SimpleNamespace(requests=[request, second], targets={"r": target, "second": second_target},
                                  manifest={"dataset_id": "fixture"})
        run = self.comparison_run()
        predictions = [{"request_id": "r", "input_fingerprint": "parent", "status": "ok",
                        "text": encode_trace(target["trace"]), "generated_tokens": 42}]
        report = evaluate(dataset, predictions, run, repetitions=10)
        self.assertEqual(report["summary"]["n"], 2)
        self.assertEqual(report["summary"]["accuracy"], .5)
        self.assertEqual(report["summary"]["valid_rate"], .5)
        self.assertEqual(report["summary"]["status_counts"], {"ok": 1, "missing": 1})
        self.assertEqual(report["summary"]["whole_trace_pass_rate"], .5)
        bad = [{**predictions[0], "text": "malformed"}]
        comparison = compare_reports(report, evaluate(dataset, bad, run, repetitions=10), repetitions=10)
        self.assertEqual(comparison["accuracy_delta"], -.5)
        changed = {**run, "lineage": {**run["lineage"], "sft_origin_sha256": "different"}}
        with self.assertRaises(ValueError):
            compare_reports(report, evaluate(dataset, bad, changed, repetitions=10), repetitions=10)
        with self.assertRaises(ValueError):
            compare_reports(report, evaluate(dataset, bad, {**run, "lineage": None}, repetitions=10), repetitions=10)
        missing_runtime = {k: v for k, v in run.items() if k != "runtime"}
        with self.assertRaises(ValueError):
            compare_reports(report, evaluate(dataset, bad, missing_runtime, repetitions=10), repetitions=10)

    def test_prediction_identity_is_enforced(self):
        request, target = example("position")
        dataset = SimpleNamespace(requests=[request], targets={"r": target}, manifest={"dataset_id": "fixture"})
        run = {"dataset_id": "fixture", "split": "train", "condition": "identity"}
        with self.assertRaises(ValueError):
            evaluate(dataset, [{"request_id": "r", "input_fingerprint": "different", "status": "ok", "text": ""}], run)

    def test_intervention_requires_same_model_and_reports_available_pairs(self):
        request, target = example("existence")
        second = {**request, "request_id": "second", "scene_id": "other", "input_fingerprint": "other"}
        dataset = SimpleNamespace(requests=[request, second], targets={"r": target, "second": {**target, "request_id": "second"}},
                                  manifest={"dataset_id": "fixture"})
        run = self.comparison_run()
        prediction = {"status": "ok", "text": encode_trace(target["trace"]), "generated_tokens": 10}
        base = evaluate(dataset, [{**prediction, "request_id": "r", "input_fingerprint": "parent"},
                                  {**prediction, "request_id": "second", "input_fingerprint": "other"}], run, repetitions=10)
        control_run = {**run, "condition": "cross_scene"}
        control = evaluate(dataset, [{**prediction, "request_id": "r", "input_fingerprint": "parent"},
            {"request_id": "second", "input_fingerprint": "other", "status": "unavailable", "text": "", "generated_tokens": 0}],
            control_run, repetitions=10)
        with self.assertRaises(ValueError): compare_reports(base, control, repetitions=10)
        result = compare_reports(base, control, repetitions=10, mode="intervention")
        self.assertEqual(result["accuracy_delta"], -.5)
        self.assertEqual(result["available_pairs"]["n"], 1)
        self.assertEqual(result["available_pairs"]["accuracy_delta"], 0.)
        control["run"]["checkpoint_sha256"] = "different"
        with self.assertRaises(ValueError): compare_reports(base, control, repetitions=10, mode="intervention")

    def test_large_finite_model_estimates_do_not_overflow_error_report(self):
        request, target = example("position")
        second = {**request, "request_id": "second", "scene_id": "other", "input_fingerprint": "other"}
        dataset = SimpleNamespace(requests=[request, second], targets={"r": target, "second": {**target, "request_id": "second"}},
                                  manifest={"dataset_id": "fixture"})
        trace = copy.deepcopy(target["trace"])
        trace["steps"][0]["value"]["x"] = 1e308
        prediction = {"status": "ok", "text": encode_trace(trace), "generated_tokens": 10}
        predictions = [{**prediction, "request_id": "r", "input_fingerprint": "parent"},
                       {**prediction, "request_id": "second", "input_fingerprint": "other"}]
        report = evaluate(dataset, predictions, {"dataset_id": "fixture", "split": "train"}, repetitions=10)
        self.assertTrue(math.isfinite(report["summary"]["numeric_errors_valid_only"]["x"]["mae"]))

    def test_grpo_ablation_requires_matching_recipe_and_budget(self):
        from b4dl_core.reasoning.evaluation import compare_training
        common = {"stage": "grpo", "recipe": {"mode": "scalar", "grpo_lr": 1e-6, "group_size": 4},
                  "step": 50, "execution": {"device": "cuda:0"}, "planned_full_budget": True}
        process = {**common, "recipe": {**common["recipe"], "mode": "process"}}
        compare_training(common, process)
        for changed in ({"step": 49}, {"execution": {"device": "cpu"}},
                        {"recipe": {**process["recipe"], "grpo_lr": 2e-6}}):
            with self.assertRaises(ValueError): compare_training(common, {**process, **changed})

    def test_free_query_matches_existing_builder_without_gt_answer(self):
        source = Path(__file__).resolve().parents[1]/"evaluation/test_b4dl.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in ("parse_frame_numbers", "lookup_ego_for_qa", "build_query")]
        namespace = {"re": re, "Optional": Optional, "Tuple": Tuple, "Dict": Dict, "VIDEO_TOKEN": "<video>",
                     "build_metatoken_line": build_metatoken_line, "render_metatoken_for_range": render_metatoken_for_range}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
        original = namespace["build_query"]
        ego = {"s": {"first_frame": "start", "last_frame": "end"},
               "s_1_3": {"first_frame": "one", "last_frame": "three"}}
        for text in ("When does it happen?", "What happens from frame 1 to frame 3?",
                     "<4DLiDAR>\n<video>\nQuestion?<meta> obsolete"):
            expected = original(text, ego_meta=ego, scene_id="s", use_4dlidar=True, use_meta=True,
                                per_sequence=True, frame_motion={}, gt="from frame 6 to frame 8",
                                answer_frames=False, task="time_grounding")
            self.assertEqual(build_free_query(text, ego, "s", {}), expected)
            self.assertNotIn("frame 6", expected)


if __name__ == "__main__":
    unittest.main()
