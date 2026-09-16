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
    build_paper_case,
    compose_paper_case,
    parse_frame_indices,
    render_bev_image,
    select_frame_indices,
    split_highlights,
)


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
