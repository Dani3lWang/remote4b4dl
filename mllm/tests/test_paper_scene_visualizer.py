import tempfile
import unittest
import json
import io
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageColor

MLLM_ROOT = Path(__file__).resolve().parents[1]
for root in (str(MLLM_ROOT), str(MLLM_ROOT / "vtimellm")):
    if root not in sys.path:
        sys.path.insert(0, root)

from paper_scene_visualizer import TARGET_PURPLE, points_in_box, render_lidar_scene
from paper_case_visualizer import (
    AnswerPanel, ReasoningCase, TemporalCase, build_paper_case, build_reasoning_board,
    compose_paper_case, PAPER_YELLOW, PAPER_GREEN,
    main,
)
from tests.test_paper_case_visualizer import _AnnotatedNuScenes, _PaperRepository


class AnnotatedRepository(_PaperRepository):
    def __init__(self):
        super().__init__()
        self.nusc = _AnnotatedNuScenes()

    def get_frame(self, scene_token, frame_index):
        frame = super().get_frame(scene_token, frame_index)
        rng = np.random.default_rng(9)
        target = np.column_stack((rng.uniform(3.1, 5.9, 500), rng.uniform(-1.9, 1.9, 500),
                                  rng.uniform(.1, 1.9, 500), np.ones(500)))
        return replace(frame, boxes=(replace(frame.boxes[0], token=f"ann-{frame_index}"),),
                       points=np.concatenate((frame.points, target)))


class SceneProjectionTests(unittest.TestCase):
    def test_oriented_membership_excludes_points_only_inside_axis_aligned_bounds(self):
        corners = np.array([[0, 2, 2, 0, 0, 2, 2, 0],
                            [0, 0, 1, 1, 0, 0, 1, 1],
                            [0, 0, 0, 0, 2, 2, 2, 2]], dtype=float)
        angle = np.pi / 4
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        rotated = rotation @ corners
        points = np.array([[1, .5, 1], [1, 1.2, 1], [1, .5, 3]]) @ rotation.T
        np.testing.assert_array_equal(points_in_box(points, rotated), [True, False, False])
        with self.assertRaisesRegex(ValueError, "尺寸无效"):
            points_in_box(points, np.zeros((3, 8)))

    def test_raw_and_highlighted_views_are_deterministic_and_do_not_mutate_xyz(self):
        frame = AnnotatedRepository().get_frame("scene-token", 0)
        original = frame.points.copy()
        raw = render_lidar_scene(frame)
        colored = render_lidar_scene(frame, target_colors={"ann-0": TARGET_PURPLE})
        self.assertEqual(raw.size, colored.size)
        purple = ImageColor.getrgb(TARGET_PURPLE)
        self.assertFalse(np.any(np.all(np.asarray(raw) == purple, axis=2)))
        self.assertGreater(np.all(np.asarray(colored) == purple, axis=2).sum(), 100)
        self.assertEqual(colored.tobytes(), render_lidar_scene(frame, target_colors={"ann-0": TARGET_PURPLE}).tobytes())
        np.testing.assert_array_equal(frame.points, original)

    def test_empty_and_nonfinite_points_produce_a_valid_image(self):
        frame = AnnotatedRepository().get_frame("scene-token", 0)
        frame = replace(frame, points=np.array([[np.nan, 0, 0, 1]]), boxes=(), tracks=())
        self.assertEqual(render_lidar_scene(frame).size, (640, 360))
        with self.assertRaisesRegex(ValueError, "有效数值"):
            render_lidar_scene(frame, elevation=float("nan"))


class FigureBoardTests(unittest.TestCase):
    def test_multicase_reasoning_exports_and_keeps_each_frame(self):
        repository = AnnotatedRepository()
        cases = [ReasoningCase("Existence", "scene-token", 0, "Is a car visible?", "Yes.",
                               ("stable-front",), ground_truth="Yes", sample_id="sample-A"),
                 ReasoningCase("3D QA", "scene-token", 4, "Where is the car?", "In front.",
                               ("stable-front",), sample_id="sample-B")]
        with tempfile.TemporaryDirectory() as directory:
            artifact = build_reasoning_board(repository, cases, output_dir=directory)
            self.assertEqual(artifact.frame_indices, (0, 4))
            self.assertEqual(artifact.image.width, 2000)
            self.assertGreater(artifact.image.height, 850)
            self.assertEqual(Path(artifact.pdf_path).read_bytes()[:5], b"%PDF-")
            with Image.open(artifact.png_path) as exported:
                self.assertEqual(exported.size, artifact.image.size)
            # Each row has a purple target and the original column has none.
            purple = ImageColor.getrgb(TARGET_PURPLE)
            pixels = np.asarray(artifact.image)
            self.assertFalse(np.any(np.all(pixels[:900, 632:1280] == purple, axis=2)))
            for y0, y1 in ((108, 450), (510, 860)):
                self.assertGreater(np.all(pixels[y0:y1, 1310:] == purple, axis=2).sum(), 50)

    def test_reasoning_rejects_absent_targets_and_out_of_range_frames(self):
        repository = AnnotatedRepository()
        case = ReasoningCase("QA", "scene-token", 0, "Q?", "A", ())
        with self.assertRaisesRegex(ValueError, "至少一个目标"):
            build_reasoning_board(repository, [case])
        with self.assertRaisesRegex(ValueError, "未出现"):
            build_reasoning_board(repository, [replace(case, target_instances=("missing",))])
        with self.assertRaisesRegex(ValueError, "超出场景"):
            build_reasoning_board(repository, [replace(case, frame_index=20, target_instances=("stable-front",))])

    def test_long_reasoning_text_grows_instead_of_being_clipped(self):
        repository = AnnotatedRepository()
        case = ReasoningCase("QA", "scene-token", 0, "Q?", "A", ("stable-front",))
        with tempfile.TemporaryDirectory() as directory:
            short = build_reasoning_board(repository, [case], output_dir=directory)
            long = build_reasoning_board(repository, [replace(case, answer="观察到车辆位于前方。" * 90)], output_dir=directory)
            self.assertGreater(long.image.height, short.image.height + 500)

    def test_temporal_cases_have_independent_questions_and_shared_evidence_colors(self):
        cell = Image.new("RGB", (640, 320), "white")
        cases = [TemporalCase("When does it move?", (AnswerPanel("Baseline A", "front rear"), AnswerPanel("Ours", "front rear"))),
                 TemporalCase("Where is it?", (AnswerPanel("Baseline B", "front rear"), AnswerPanel("Ours", "front rear")))]
        figure = compose_paper_case([cell] * 2, [cell] * 2, [cell] * 2, [0, 1], "", "", "",
                                    comparison_cases=cases, yellow_phrases=("front",), green_phrases=("rear",))
        pixels = np.asarray(figure)
        for x0, x1 in ((32, 995), (1005, 1968)):
            for color in (PAPER_YELLOW, PAPER_GREEN):
                self.assertGreater(np.all(pixels[736:-68, x0:x1] == ImageColor.getrgb(color), axis=2).sum(), 100)
        with self.assertRaisesRegex(ValueError, "两组模型"):
            compose_paper_case([cell] * 2, [cell] * 2, [cell] * 2, [0, 1], "", "", "",
                               comparison_cases=[replace(cases[0], answers=cases[0].answers[:1])])

    def test_temporal_renderer_supports_3d_and_legacy_bev_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            for view in ("3d", "bev"):
                artifact = build_paper_case(AnnotatedRepository(), "scene-token", [0, 4], "Q?", "B", "O",
                                            front_instance="stable-front", lidar_view=view, output_dir=directory)
                self.assertEqual(artifact.image.width, 2000)

    def test_cli_reads_both_case_formats_without_single_case_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            for layout, rows in (
                ("reasoning", [{"task_label": "QA", "scene_token": "scene-token", "frame_index": 0,
                                "question": "Q?", "answer": "A", "target_instances": ["stable-front"]}]),
                ("comparison", [{"question": "Q?", "answers": [{"label": "Baseline", "answer": "B"},
                                                               {"label": "Ours", "answer": "O"}]}]),
            ):
                with self.subTest(layout=layout):
                    path = Path(directory) / f"{layout}.json"
                    path.write_text(json.dumps(rows), encoding="utf-8")
                    with patch("paper_case_visualizer.NuScenesSceneRepository", return_value=AnnotatedRepository()):
                        with patch("sys.stdout", new_callable=io.StringIO) as output:
                            main(["--nuscenes-root", directory, "--layout", layout, "--cases-json", str(path),
                                  "--scene-token", "scene-token", "--frames", "0,4", "--output-dir", directory])
                    png, pdf = output.getvalue().strip().splitlines()
                    self.assertTrue(Path(png).is_file())
                    self.assertEqual(Path(pdf).read_bytes()[:5], b"%PDF-")


if __name__ == "__main__":
    unittest.main()
