import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
from PIL import Image


MLLM_ROOT = Path(__file__).resolve().parents[1]
VTIMELLM_ROOT = MLLM_ROOT / "vtimellm"
for root in (str(VTIMELLM_ROOT), str(MLLM_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from lidar_visualizer import BoxRender, FrameData, SceneRef, TrackRender
from paper_case_visualizer import (
    AnswerPanel,
    PAPER_GREEN,
    PAPER_YELLOW,
    _fit_cell,
    _font,
    _wrapped_lines,
    build_paper_case,
    compose_paper_case,
    parse_frame_indices,
    render_bev_image,
    render_annotated_camera,
    scene_target_choices,
    select_frame_indices,
    split_highlights,
)
from PIL import ImageDraw, ImageColor


class _AnnotatedNuScenes:
    def get(self, table, token):
        if table == "sample":
            index = token.rsplit("-", 1)[-1]
            return {"data": {"CAM_FRONT": "camera"}, "anns": [f"ann-{index}"]}
        if table == "sample_annotation":
            return {"instance_token": "stable-front", "category_name": "vehicle.car"}
        raise KeyError((table, token))

    @staticmethod
    def get_sample_data(token):
        class CameraBox:
            token = "ann-0"
            name = "vehicle.car"

            @staticmethod
            def corners():
                return np.array([[-1, 1, 1, -1, -1, 1, 1, -1],
                                 [-1, -1, 1, 1, -1, -1, 1, 1],
                                 [5, 5, 5, 5, 7, 7, 7, 7]], dtype=float)
        return "unused", [CameraBox()], np.array([[200, 0, 320], [0, 200, 160], [0, 0, 1]])


class _NoCameraNuScenes:
    @staticmethod
    def get(table, token):
        if table == "sample":
            return {"data": {}}
        raise KeyError((table, token))


class _PaperRepository:
    def __init__(self):
        self.scene = SceneRef(
            "scene-token", "000001", "scene-demo", "val",
            tuple(f"sample-{index}" for index in range(9)),
        )
        self.nusc = _NoCameraNuScenes()

    def get_scene(self, scene_token):
        if scene_token != self.scene.scene_token:
            raise KeyError(scene_token)
        return self.scene

    def get_frame(self, scene_token, frame_index):
        self.get_scene(scene_token)
        corners = np.array(
            [
                [6, 6, 3, 3, 6, 6, 3, 3],
                [2, -2, -2, 2, 2, -2, -2, 2],
                [0, 0, 0, 0, 2, 2, 2, 2],
            ],
            dtype=np.float32,
        )
        x = np.linspace(-20, 20, 240, dtype=np.float32)
        points = np.column_stack((x, np.sin(x) * 8, np.cos(x), np.ones_like(x)))
        return FrameData(
            scene=self.scene,
            frame_index=int(frame_index),
            sample_token=f"sample-{frame_index}",
            points=points,
            boxes=(BoxRender("box", "vehicle.car", corners),),
            tracks=(
                TrackRender(
                    "box", "vehicle.car",
                    np.array([[1, 0, 0], [3, 0.5, 0], [5, 1, 0]], dtype=np.float32),
                ),
            ),
            camera_paths={},
        )

    @staticmethod
    def camera_image(frame, camera):
        color = "#637f92" if camera == "CAM_FRONT" else "#8b7566"
        return Image.new("RGB", (640, 320), color)


class FrameSelectionTests(unittest.TestCase):
    def test_evenly_selects_five_frames_including_scene_ends(self):
        self.assertEqual(select_frame_indices(9), (0, 2, 4, 6, 8))
        with self.assertRaisesRegex(ValueError, "至少需要 2"):
            select_frame_indices(1)

    def test_explicit_frames_are_sorted_and_validated(self):
        self.assertEqual(parse_frame_indices("8, 0, 4", 9), (0, 4, 8))
        with self.assertRaisesRegex(ValueError, "不能重复"):
            parse_frame_indices("0, 0, 4", 9)
        with self.assertRaisesRegex(ValueError, "超出"):
            parse_frame_indices("0, 4, 9", 9)

    def test_highlight_lists_accept_chinese_and_ascii_separators(self):
        self.assertEqual(
            split_highlights("vehicles in front，后方车辆; turning left"),
            ("vehicles in front", "后方车辆", "turning left"),
        )


class PaperRenderingTests(unittest.TestCase):
    def test_target_instances_are_stable_across_annotation_tokens(self):
        repository = _PaperRepository()
        repository.nusc = _AnnotatedNuScenes()
        self.assertEqual(scene_target_choices(repository, "scene-token")[0][1], "stable-front")
        with tempfile.TemporaryDirectory() as directory:
            artifact = build_paper_case(
                repository, "scene-token", (0, 4, 8), "Question?", "Baseline", "Ours",
                front_instance="stable-front", show_boxes=False, output_dir=directory,
            )
            self.assertEqual(artifact.frame_indices, (0, 4, 8))
            with self.assertRaisesRegex(ValueError, "未出现"):
                build_paper_case(repository, "scene-token", (0, 4), "Q", "B", "O",
                                 rear_instance="missing", output_dir=directory)
            with self.assertRaisesRegex(ValueError, "同一个"):
                build_paper_case(repository, "scene-token", (0, 4), "Q", "B", "O",
                                 front_instance="stable-front", rear_instance="stable-front")

    def test_selected_boxes_keep_the_same_color_in_camera_and_bev(self):
        repository = _PaperRepository()
        repository.nusc = _AnnotatedNuScenes()
        frame = repository.get_frame("scene-token", 0)
        camera = render_annotated_camera(repository, frame, "CAM_FRONT", False,
                                         {"ann-0": PAPER_YELLOW})
        bev = render_bev_image(frame, show_boxes=False, target_colors={"box": PAPER_YELLOW})
        yellow = ImageColor.getrgb(PAPER_YELLOW)
        self.assertGreater(sum(n for n, c in camera.getcolors(100_000) if c == yellow), 100)
        self.assertGreater(sum(n for n, c in bev.getcolors(100_000) if c == yellow), 20)

    def test_full_field_of_view_is_preserved(self):
        source = Image.new("RGB", (240, 80), "blue")
        ImageDraw.Draw(source).rectangle((230, 0, 239, 79), fill="red")
        cell = _fit_cell(source, (100, 100))
        self.assertGreater(sum(n for n, c in cell.getcolors(100_000) if c == (255, 0, 0)), 0)

    def test_semantic_colors_are_shared_by_both_models(self):
        cell = Image.new("RGB", (640, 320), "white")
        figure = compose_paper_case([cell] * 2, [cell] * 2, [cell] * 2, [0, 1],
                                    "Question?", "front and rear", "front and rear",
                                    yellow_phrases=("front",), green_phrases=("rear",))
        # Above the legend, each model panel contains both evidence colors.
        for x0, x1 in ((32, 995), (1005, 1968)):
            colors = dict((color, count) for count, color in
                          figure.crop((x0, 840, x1, figure.height - 64)).getcolors(100_000))
            self.assertGreater(colors.get(ImageColor.getrgb(PAPER_YELLOW), 0), 100)
            self.assertGreater(colors.get(ImageColor.getrgb(PAPER_GREEN), 0), 100)

    def test_ablation_and_long_multilingual_answers_grow_without_truncation(self):
        cell = Image.new("RGB", (640, 320), "white")
        short = [AnswerPanel("No HA", "front"), AnswerPanel("No meta", "front"), AnswerPanel("Ours", "rear")]
        kwargs = dict(front_images=[cell] * 2, back_images=[cell] * 2, bev_images=[cell] * 2,
                      frame_indices=[0, 1], question="Q", baseline_answer="", b4dl_answer="",
                      layout="ablation", ground_truth="Reference answer")
        compact = compose_paper_case(**kwargs, answer_panels=short)
        long_text = "中文观察与证据，" * 80 + "\n" + "x" * 200
        expanded = compose_paper_case(**kwargs, answer_panels=[*short[:2], AnswerPanel("Ours", long_text)])
        self.assertGreater(expanded.height, compact.height + 500)
        spans = _wrapped_lines(ImageDraw.Draw(cell), long_text, _font(23), 400)
        recovered = "".join(long_text[a:b] for a, b in spans)
        self.assertEqual(recovered, long_text.replace("\n", ""))
        with self.assertRaisesRegex(ValueError, "3 组"):
            compose_paper_case(**kwargs, answer_panels=short[:2])
        with self.assertRaisesRegex(ValueError, "实际答案"):
            compose_paper_case(**kwargs, answer_panels=[*short[:2], AnswerPanel("Ours", "")])

    def test_bev_renderer_draws_points_boxes_and_tracks(self):
        repository = _PaperRepository()
        image = render_bev_image(repository.get_frame("scene-token", 0), size=(500, 260))
        self.assertEqual(image.size, (500, 260))
        self.assertGreater(len(image.getcolors(maxcolors=100_000)), 8)

    def test_composer_requires_synchronized_rows(self):
        image = Image.new("RGB", (320, 180), "white")
        with self.assertRaisesRegex(ValueError, "数量一致"):
            compose_paper_case(
                [image, image], [image], [image, image], [0, 1],
                "Question?", "Baseline", "Ours",
            )

    def test_build_exports_png_and_pdf(self):
        repository = _PaperRepository()
        with tempfile.TemporaryDirectory() as directory:
            artifact = build_paper_case(
                repository=repository,
                scene_token="scene-token",
                frame_indices=(0, 2, 4, 6, 8),
                question="What dynamic movement is observed throughout the frames?",
                baseline_answer="Vehicles in front move forward.",
                b4dl_answer="Vehicles in front move forward while rear vehicles move away.",
                baseline_highlights=("Vehicles in front",),
                b4dl_highlights=("rear vehicles",),
                output_dir=directory,
            )
            self.assertEqual(artifact.frame_indices, (0, 2, 4, 6, 8))
            self.assertEqual(artifact.image.width, 2000)
            self.assertTrue(Path(artifact.png_path).is_file())
            self.assertTrue(Path(artifact.pdf_path).is_file())
            with Image.open(artifact.png_path) as rendered:
                self.assertEqual(rendered.size, artifact.image.size)


if __name__ == "__main__":
    unittest.main()
