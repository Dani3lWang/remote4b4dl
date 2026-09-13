import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MLLM_ROOT = Path(__file__).resolve().parents[1]
VTIMELLM_ROOT = MLLM_ROOT / "vtimellm"
for root in (str(VTIMELLM_ROOT), str(MLLM_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from demo_gradio import OptionalInferenceEngine, _model_option_state
from lidar_visualizer import (
    BoxRender,
    FrameData,
    NuScenesSceneRepository,
    SceneRef,
    TrackRender,
    advance_playback,
    box_line_coordinates,
    category_group,
    deterministic_downsample,
    global_to_sensor,
    make_plotly_figures,
    step_frame,
    validate_scene_features,
)


class FakeNuScenes:
    def __init__(self):
        self.scene = [
            {
                "token": "scene-token",
                "name": "scene-0001",
                "first_sample_token": "sample-0",
            }
        ]
        self.records = {
            ("sample", "sample-0"): {
                "token": "sample-0",
                "next": "sample-1",
                "data": {"LIDAR_TOP": "lidar-0"},
            },
            ("sample", "sample-1"): {
                "token": "sample-1",
                "next": "",
                "data": {"LIDAR_TOP": "lidar-1"},
            },
            ("sample_data", "lidar-0"): {
                "filename": "samples/LIDAR_TOP/0.bin",
                "ego_pose_token": "pose-0",
                "calibrated_sensor_token": "calib",
            },
            ("sample_data", "lidar-1"): {
                "filename": "samples/LIDAR_TOP/1.bin",
                "ego_pose_token": "pose-1",
                "calibrated_sensor_token": "calib",
            },
            ("ego_pose", "pose-0"): {
                "translation": [0, 0, 0],
                "rotation": [1, 0, 0, 0],
            },
            ("ego_pose", "pose-1"): {
                "translation": [0, 0, 0],
                "rotation": [1, 0, 0, 0],
            },
            ("calibrated_sensor", "calib"): {
                "translation": [0, 0, 0],
                "rotation": [1, 0, 0, 0],
            },
            ("sample_annotation", "ann-current"): {
                "translation": [1, 0, 0],
                "prev": "ann-previous",
            },
            ("sample_annotation", "ann-previous"): {
                "translation": [0, 0, 0],
                "prev": "",
            },
        }

    def get(self, table, token):
        return self.records[(table, token)]

    def get_sample_data(self, lidar_token):
        return "unused.bin", [FakeBox()], None


class FakeBox:
    token = "ann-current"
    name = "vehicle.car"

    @staticmethod
    def corners():
        return np.arange(24, dtype=np.float32).reshape(3, 8)


class MissingAnnotationNuScenes(FakeNuScenes):
    def get_sample_data(self, lidar_token):
        raise RuntimeError("annotations unavailable")


class SyntheticRepository(NuScenesSceneRepository):
    def _load_points(self, lidar_path, sample_token):
        return np.array([[1.0, 2.0, 0.5, 0.8]], dtype=np.float32)

class GeometryTests(unittest.TestCase):
    def test_deterministic_downsample_is_stable_and_bounded(self):
        points = np.arange(400, dtype=np.float32).reshape(100, 4)
        first = deterministic_downsample(points, 12, "sample-a")
        second = deterministic_downsample(points, 12, "sample-a")
        self.assertEqual(first.shape, (12, 4))
        np.testing.assert_array_equal(first, second)

    def test_global_to_sensor_applies_both_translations(self):
        points = np.array([[12.0, 5.0, 1.0]], dtype=np.float64)
        identity = [1.0, 0.0, 0.0, 0.0]
        transformed = global_to_sensor(
            points,
            {"translation": [10.0, 2.0, 0.0], "rotation": identity},
            {"translation": [1.0, 1.0, 0.0], "rotation": identity},
        )
        np.testing.assert_allclose(transformed, [[1.0, 2.0, 1.0]])

    def test_global_to_sensor_respects_ego_rotation(self):
        half = np.sqrt(0.5)
        transformed = global_to_sensor(
            np.array([[0.0, 1.0, 0.0]]),
            {"translation": [0, 0, 0], "rotation": [half, 0, 0, half]},
            {"translation": [0, 0, 0], "rotation": [1, 0, 0, 0]},
        )
        np.testing.assert_allclose(transformed, [[1.0, 0.0, 0.0]], atol=1e-6)

    def test_box_lines_include_all_twelve_edges(self):
        corners = np.arange(24, dtype=np.float32).reshape(3, 8)
        lines = box_line_coordinates(corners)
        self.assertEqual(lines.shape, (36, 3))
        self.assertEqual(np.isnan(lines[:, 0]).sum(), 12)

    def test_categories_have_stable_groups(self):
        self.assertEqual(category_group("vehicle.car"), "vehicle")
        self.assertEqual(category_group("vehicle.bicycle"), "cycle")
        self.assertEqual(category_group("human.pedestrian.adult"), "pedestrian")

    def test_plotly_3d_and_bev_include_boxes_and_tracks(self):
        scene = SceneRef("scene", "001", "scene-1", "val", ("sample",))
        corners = np.array(
            [
                [1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1],
                [0, 0, 0, 0, 2, 2, 2, 2],
            ],
            dtype=np.float32,
        )
        frame = FrameData(
            scene=scene,
            frame_index=0,
            sample_token="sample",
            points=np.array([[0, 0, 0, 1], [2, 2, 1, 0.5]], dtype=np.float32),
            boxes=(BoxRender("ann", "vehicle.car", corners),),
            tracks=(TrackRender("ann", "vehicle.car", np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)),),
            camera_paths={},
        )
        figure_3d, figure_bev = make_plotly_figures(frame, True, True)
        self.assertEqual(len(figure_3d.data), 3)
        self.assertEqual(len(figure_bev.data), 3)


class RepositoryTests(unittest.TestCase):
    def test_scene_metadata_mapping_and_lazy_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "scene_metadata.json"
            metadata.write_text(
                json.dumps(
                    [{
                        "scene_token": "scene-token",
                        "scene_id": "003833660",
                        "split": "val",
                    }]
                ),
                encoding="utf-8",
            )
            repository = SyntheticRepository(
                dataroot=directory,
                scene_metadata=str(metadata),
                nusc=FakeNuScenes(),
            )
            scene = repository.scenes[0]
            self.assertEqual(scene.scene_id, "003833660")
            self.assertEqual(scene.sample_tokens, ("sample-0", "sample-1"))
            frame = repository.get_frame(scene.scene_token, 0)
            self.assertEqual(frame.points.shape, (1, 4))
            self.assertEqual(len(frame.boxes), 1)
            self.assertEqual(len(frame.tracks), 1)
            np.testing.assert_allclose(frame.tracks[0].points, [[0, 0, 0], [1, 0, 0]])
            self.assertTrue(all(path is None for path in frame.camera_paths.values()))
            self.assertIs(repository.get_frame(scene.scene_token, 0), frame)

    def test_timeline_clamps_at_both_ends(self):
        self.assertEqual(step_frame(0, 40, -1), 0)
        self.assertEqual(step_frame(39, 40, 1), 39)
        self.assertEqual(step_frame(12, 40, 1), 13)

    def test_playback_stops_on_last_frame(self):
        self.assertEqual(advance_playback(38, 40), (39, True))
        self.assertEqual(advance_playback(39, 40), (39, False))
        self.assertEqual(advance_playback(0, 0), (0, False))

    def test_missing_annotations_degrade_to_points_only(self):
        repository = SyntheticRepository(
            dataroot=str(MLLM_ROOT), nusc=MissingAnnotationNuScenes()
        )
        frame = repository.get_frame("scene-token", 0)
        self.assertEqual(frame.boxes, ())
        self.assertEqual(frame.tracks, ())
        self.assertIn("标注不可用", frame.warnings[0])
        placeholder = repository.camera_image(frame, "CAM_FRONT")
        self.assertEqual(placeholder.size, (960, 540))


class FeatureAndModeTests(unittest.TestCase):
    def test_feature_shape_and_frame_alignment_are_strict(self):
        validate_scene_features(np.zeros((40, 768), dtype=np.float16), 40)
        with self.assertRaisesRegex(ValueError, "特征帧数"):
            validate_scene_features(np.zeros((39, 768), dtype=np.float16), 40)
        with self.assertRaisesRegex(ValueError, r"\[N, 768\]"):
            validate_scene_features(np.zeros((40, 512), dtype=np.float16), 40)

    def test_viewer_mode_does_not_import_or_require_model(self):
        args = argparse.Namespace(
            model_base=None,
            pretrain_mm_mlp_adapter=None,
            stage2=None,
            stage3=None,
            feat_folder=None,
            gpu_id=0,
        )
        enabled, missing = _model_option_state(args)
        self.assertFalse(enabled)
        self.assertEqual(missing, [])
        engine = OptionalInferenceEngine(args)
        self.assertFalse(engine.enabled)

    def test_partial_model_configuration_is_rejected(self):
        args = argparse.Namespace(
            model_base="base",
            pretrain_mm_mlp_adapter=None,
            stage2=None,
            stage3=None,
            feat_folder=None,
            gpu_id=0,
        )
        enabled, missing = _model_option_state(args)
        self.assertFalse(enabled)
        self.assertIn("stage2", missing)
        with self.assertRaisesRegex(ValueError, "成套提供"):
            OptionalInferenceEngine(args)


if __name__ == "__main__":
    unittest.main()
