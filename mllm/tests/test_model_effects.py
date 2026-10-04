import json
import argparse
import sys
import tempfile
import unittest
from pathlib import Path


MLLM_ROOT = Path(__file__).resolve().parents[1]
VTIMELLM_ROOT = MLLM_ROOT / "vtimellm"
for root in (str(VTIMELLM_ROOT), str(MLLM_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from model_effects import (  # noqa: E402
    EvaluationRepository,
    interval_iou,
    normalize_answer,
    parse_frame_interval,
    rouge_l_f1,
)


def _write_json(directory: str, name: str, value) -> str:
    path = Path(directory) / name
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return str(path)


class MetricPrimitiveTests(unittest.TestCase):
    def test_answer_normalization_matches_evaluator_rules(self):
        self.assertEqual(normalize_answer("The answer is Yes."), "yes")
        self.assertEqual(normalize_answer("A: No!"), "no")

    def test_time_interval_parser_and_closed_iou(self):
        self.assertEqual(parse_frame_interval("from frame 006 to frame 014"), (6, 14))
        self.assertEqual(parse_frame_interval("frame 8 until frame 9"), None)
        self.assertAlmostEqual(interval_iou((6, 14), (10, 18)), 5 / 13)
        self.assertEqual(interval_iou(None, (1, 2)), 0.0)
        self.assertEqual(interval_iou((4, 2), (1, 3)), 0.0)

    def test_lightweight_rouge_l_is_bounded(self):
        self.assertEqual(rouge_l_f1("car turns left", "car turns left"), 1.0)
        self.assertEqual(rouge_l_f1("car", "pedestrian"), 0.0)
        self.assertGreater(rouge_l_f1("car turns", "the car turns left"), 0.5)


class RepositoryTests(unittest.TestCase):
    def test_loads_v2_metadata_and_scores_samples(self):
        predictions = {
            "_schema_version": 2,
            "_run": {"model": "b3"},
            "existence": {
                "predictions": ["Yes."],
                "ground_truths": ["Yes."],
                "questions": ["Was a car present?"],
                "samples": [{
                    "sample_id": "existence:000007",
                    "source_index": 7,
                    "scene_id": "scene-7",
                }],
            },
            "time_grounding": {
                "predictions": ["from frame 15 to frame 25"],
                "ground_truths": ["from frame 18 to frame 27"],
                "questions": ["When did it turn?"],
                "samples": [{"source_index": 3, "scene_id": "scene-3"}],
            },
        }
        metrics = {"final_scores": {"accuracy": 1.0, "miou": 8 / 13}}
        with tempfile.TemporaryDirectory() as directory:
            repo = EvaluationRepository.from_files(
                _write_json(directory, "predictions.json", predictions),
                _write_json(directory, "metrics.json", metrics),
            )
        self.assertEqual(len(repo.samples), 2)
        existence = repo.get("existence:000007")
        self.assertEqual(existence.status, "correct")
        self.assertEqual(existence.score, 1.0)
        tg = repo.get("time_grounding:000003")
        self.assertEqual(tg.default_frame, 15)
        self.assertAlmostEqual(tg.score, 8 / 13)
        self.assertEqual(repo.final_scores["accuracy"], 1.0)

    def test_legacy_results_are_linked_only_by_question_and_gt(self):
        predictions = {
            "existence": {
                "predictions": ["No."],
                "ground_truths": ["No."],
                "questions": ["Was a bus present in frame 006?"],
            }
        }
        test_data = [{
            "scene_id": "005745653",
            "scene_token": "scene-token",
            "task": "existence",
            "conversations": [
                {"from": "human", "value": "<4DLiDAR>\n<video>\nWas a bus present in frame 006?"},
                {"from": "gpt", "value": "No."},
            ],
        }]
        with tempfile.TemporaryDirectory() as directory:
            repo = EvaluationRepository.from_files(
                _write_json(directory, "legacy.json", predictions),
                test_data_path=_write_json(directory, "test.json", test_data),
            )
        self.assertEqual(repo.samples[0].scene_id, "005745653")
        self.assertEqual(repo.samples[0].scene_token, "scene-token")
        self.assertEqual(repo.samples[0].default_frame, 6)
        self.assertEqual(repo.warnings, ())

    def test_unmatched_legacy_result_remains_unlinked(self):
        predictions = {
            "existence": {
                "predictions": ["Yes."],
                "ground_truths": ["Yes."],
                "questions": ["Different question"],
            }
        }
        test_data = [{
            "scene_id": "wrong-scene",
            "task": "existence",
            "conversations": [
                {"from": "human", "value": "Original question"},
                {"from": "gpt", "value": "Yes."},
            ],
        }]
        with tempfile.TemporaryDirectory() as directory:
            repo = EvaluationRepository.from_files(
                _write_json(directory, "legacy.json", predictions),
                test_data_path=_write_json(directory, "test.json", test_data),
            )
        self.assertIsNone(repo.samples[0].scene_id)
        self.assertTrue(repo.warnings)

    def test_repeated_legacy_qa_across_scenes_is_never_linked_by_order(self):
        predictions = {"existence": {
            "predictions": ["Yes."], "ground_truths": ["Yes."], "questions": ["Car?"],
        }}
        test_data = [
            {"task": "existence", "scene_id": scene,
             "conversations": [{"value": "Car?"}, {"value": "Yes."}]}
            for scene in ("scene-a", "scene-b")
        ]
        with tempfile.TemporaryDirectory() as directory:
            repo = EvaluationRepository.from_files(
                _write_json(directory, "p.json", predictions),
                test_data_path=_write_json(directory, "test.json", test_data),
            )
        self.assertIsNone(repo.samples[0].scene_id)
        self.assertTrue(repo.warnings)

    def test_repeated_legacy_qa_with_different_input_frames_is_ambiguous(self):
        predictions = {"existence": {
            "predictions": ["Yes."], "ground_truths": ["Yes."], "questions": ["Car?"],
        }}
        test_data = [
            {"task": "existence", "scene_id": "same-scene", "feat_indices": frames,
             "conversations": [{"value": "Car?"}, {"value": "Yes."}]}
            for frames in ([0, 1], [2, 3])
        ]
        with tempfile.TemporaryDirectory() as directory:
            repo = EvaluationRepository.from_files(
                _write_json(directory, "p.json", predictions),
                test_data_path=_write_json(directory, "test.json", test_data),
            )
        self.assertIsNone(repo.samples[0].scene_id)

    def test_oracle_and_meteor_provenance_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = EvaluationRepository.from_files(
                _write_json(directory, "p.json", {"_run": {"answer_frames": True}}),
                _write_json(directory, "m.json", {
                    "metric_backend": {"meteor": {"reported": "nltk-meteor-1.0"}},
                }),
            )
        self.assertIn("oracle", " ".join(repo.warnings))
        self.assertIn("nltk-meteor-1.0", " ".join(repo.warnings))

    def test_empty_non_object_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(directory, "bad.json", [])
            with self.assertRaisesRegex(ValueError, "顶层必须是对象"):
                EvaluationRepository.from_files(path)

    def test_misaligned_arrays_fail_fast(self):
        predictions = {
            "binary_qa": {
                "predictions": ["Yes."],
                "ground_truths": [],
                "questions": ["Question"],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = _write_json(directory, "bad.json", predictions)
            with self.assertRaisesRegex(ValueError, "数组长度不一致"):
                EvaluationRepository.from_files(path)

    def test_filter_and_page(self):
        predictions = {
            "existence": {
                "predictions": ["Yes.", "No."],
                "ground_truths": ["No.", "No."],
                "questions": ["car present", "bus present"],
                "samples": [
                    {"source_index": 0, "scene_id": "a"},
                    {"source_index": 1, "scene_id": "b"},
                ],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            repo = EvaluationRepository.from_files(_write_json(directory, "p.json", predictions))
        values, total, page = repo.page(
            task="existence", status="error", query="car", page_size=1
        )
        self.assertEqual(total, 1)
        self.assertEqual(page, 0)
        self.assertEqual(values[0].scene_id, "a")


class FigureTests(unittest.TestCase):
    def test_timeline_uses_dataset_frame_indices(self):
        try:
            from effect_visualizer import timeline_figure
        except ImportError as exc:
            self.skipTest(str(exc))
        predictions = {
            "time_grounding": {
                "predictions": ["from frame 6 to frame 14"],
                "ground_truths": ["from frame 8 to frame 16"],
                "questions": ["When?"],
                "samples": [{"source_index": 0}],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            repo = EvaluationRepository.from_files(_write_json(directory, "p.json", predictions))
        figure = timeline_figure(repo.samples[0], 40, 6)
        self.assertEqual(list(figure.data[0].x), [8, 16])
        self.assertEqual(list(figure.data[1].x), [6, 14])


class UIConstructionTests(unittest.TestCase):
    def test_effect_dashboard_builds_without_model_or_cuda(self):
        try:
            import gradio  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))
        from demo_gradio import OptionalInferenceEngine, create_demo
        from tests.test_lidar_visualizer import FakeNuScenes, SyntheticRepository

        predictions = {
            "time_grounding": {
                "predictions": ["from frame 0 to frame 1"],
                "ground_truths": ["from frame 0 to frame 1"],
                "questions": ["When was the event visible?"],
                "samples": [{"source_index": 0, "scene_id": "003833660"}],
            }
        }
        args = argparse.Namespace(
            model_base=None,
            pretrain_mm_mlp_adapter=None,
            stage2=None,
            stage3=None,
            feat_folder=None,
            gpu_id=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "scene_metadata.json"
            metadata.write_text(
                json.dumps([{
                    "scene_token": "scene-token",
                    "scene_id": "003833660",
                    "split": "val",
                }]),
                encoding="utf-8",
            )
            repository = SyntheticRepository(
                dataroot=directory,
                scene_metadata=str(metadata),
                nusc=FakeNuScenes(),
            )
            effects = EvaluationRepository.from_files(
                _write_json(directory, "predictions.json", predictions)
            )
            demo = create_demo(repository, OptionalInferenceEngine(args), effects)
        self.assertTrue(callable(getattr(demo, "queue", None)))

    def test_unlinked_sample_does_not_export_with_default_scene(self):
        import gradio  # noqa: F401
        from demo_gradio import OptionalInferenceEngine, create_demo
        from model_effects import EvaluationSample
        from tests.test_lidar_visualizer import FakeNuScenes, SyntheticRepository

        args = argparse.Namespace(model_base=None, pretrain_mm_mlp_adapter=None,
                                  stage2=None, stage3=None, feat_folder=None)
        with tempfile.TemporaryDirectory() as directory:
            repository = SyntheticRepository(dataroot=directory, nusc=FakeNuScenes())
            sample = EvaluationSample("unlinked", "existence", 0, "Car?", "Yes", "No")
            demo = create_demo(repository, OptionalInferenceEngine(args), EvaluationRepository([sample]))
            callback = next(fn.fn for fn in demo.fns.values()
                            if fn.fn is not None and fn.fn.__name__ == "load_paper_sample")
            loaded = callback("unlinked")
        self.assertIsNone(loaded[0]["value"])
        self.assertEqual(loaded[1], "")
        self.assertIn("手动", loaded[5])
        self.assertEqual(loaded[3], sample.ground_truth)
        self.assertEqual(loaded[6:], ("",) * 8 + (None, None))

    def test_paper_export_requires_real_model_answers_and_separates_ground_truth(self):
        from demo_gradio import OptionalInferenceEngine, create_demo
        from model_effects import EvaluationSample
        from tests.test_lidar_visualizer import FakeNuScenes, SyntheticRepository

        args = argparse.Namespace(model_base=None, pretrain_mm_mlp_adapter=None,
                                  stage2=None, stage3=None, feat_folder=None)
        with tempfile.TemporaryDirectory() as directory:
            repository = SyntheticRepository(dataroot=directory, nusc=FakeNuScenes())
            sample = EvaluationSample("linked", "existence", 0, "Car?", "Yes", "No",
                                      scene_token="scene-token")
            demo = create_demo(repository, OptionalInferenceEngine(args), EvaluationRepository([sample]))
            callbacks = {fn.fn.__name__: fn.fn for fn in demo.fns.values() if fn.fn is not None}
            loaded = callbacks["load_paper_sample"]("linked")
            self.assertEqual(loaded[3:5], ("Yes", "No"))
            self.assertEqual(loaded[6], "")  # Ground Truth must not impersonate a baseline.
            inputs = ["scene-token", "0,1", "Case", "Car?", "Baseline", "", "",
                      "B4DL", "No", "", True, True, "comparison", "Yes",
                      "Ablation", "", "", "", "", None, None, ""]
            failed = callbacks["export_paper_case"](*inputs)
            self.assertIsNone(failed[0])
            self.assertIn("实际答案", failed[2])
            inputs[5] = "Yes"
            rendered = callbacks["export_paper_case"](*inputs)
            self.assertEqual(rendered[0].width, 2000)
            self.assertEqual(len(rendered[1]), 2)
            self.assertIn("导出完成", rendered[2])
            changed = callbacks["change_paper_layout"]("ablation")
            self.assertEqual(changed[1], "B4DL without HA and Metatoken")
            self.assertEqual(changed[2:], ("",) * 5 + (None, None))


if __name__ == "__main__":
    unittest.main()
