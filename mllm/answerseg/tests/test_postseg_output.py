"""Free text -> segmentation -> output markers and exact mask-row association."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from answerseg.language import make_prompt, training_sequence
from answerseg.output import answer_spans, render_segmented_answer
from answerseg.runner import save_prediction, score_prediction
from test_answerseg import example_fixture, model_fixture


class PostSegmentationTests(unittest.TestCase):
    def test_plain_free_answer_segments_before_rendering_marker(self):
        model = model_fixture().eval()
        prompt = make_prompt("Which truck is ahead?")
        ids, _ = training_sequence(model.bridge.tokenizer, prompt, "The truck ahead.")
        events = []

        def generate(**kwargs):
            events.append("free_answer")
            self.assertFalse(ids.eq(model.tokens.seg).any())
            return ids

        def select(states, queries):
            events.append("segmentation")
            return torch.tensor([[20.0, -20.0, -20.0]])

        def render(*args):
            events.append("render")
            return render_segmented_answer(*args)

        with (
            patch.object(model.language, "generate", side_effect=generate),
            patch.object(model.head, "forward", side_effect=select),
            patch("answerseg.model.render_segmented_answer", side_effect=render),
        ):
            result = model.predict(
                prompt, torch.randn(3, 768), torch.zeros(4, 4), max_new_tokens=16
            )
        self.assertEqual(events, ["free_answer", "segmentation", "render"])
        self.assertEqual(result["answer"].count("<SEG>"), 1)
        self.assertNotIn("<SEG>", result["plain_answer"])
        self.assertEqual(result["answer"].replace(" <SEG>", ""), result["plain_answer"])
        self.assertEqual(result["masks"][0].tolist(), [True, True, False, False])
        self.assertEqual(result["objects"][0]["mask_index"], 0)
        self.assertEqual(result["seg_token_source"], "post_segmentation")
        start, end = result["objects"][0]["output_marker_span"]
        self.assertEqual(result["answer"][start:end], "<SEG>")

    def test_rejection_and_low_confidence_preserve_text_without_marker(self):
        model = model_fixture().eval()
        for logits, reason in (
            ([0.0, 0.0, 0.0], "low_binding_confidence"),
            ([-20.0, -20.0, 20.0], "predicted_no_referent"),
        ):
            with patch.object(
                model.head, "forward", return_value=torch.tensor([logits])
            ):
                result = model.segment_answer(
                    make_prompt("truck"),
                    torch.randn(3, 768),
                    torch.zeros(4, 4),
                    "The truck ahead.",
                )
            self.assertEqual(result["answer"], "The truck ahead.")
            self.assertEqual(result["masks"], [])
            self.assertEqual(result["objects"][0]["status"], reason)
            self.assertIsNone(result["objects"][0]["mask_index"])
            self.assertIsNone(result["objects"][0]["output_marker_span"])

    def test_failed_first_span_does_not_shift_second_mask_in_output_or_metrics(self):
        model = model_fixture().eval()
        with patch.object(
            model.head,
            "forward",
            return_value=torch.tensor([[0.0, 0.0, 0.0], [-20.0, 20.0, -20.0]]),
        ):
            result = model.segment_answer(
                make_prompt("truck"),
                torch.randn(3, 768),
                torch.zeros(4, 4),
                "truck and car",
                spans=[[0, 5], [10, 13]],
            )
        self.assertEqual(result["answer"], "truck and car <SEG>")
        self.assertEqual(len(result["masks"]), 1)
        self.assertIsNone(result["objects"][0]["seg_index"])
        self.assertEqual(result["objects"][1]["seg_index"], 0)
        self.assertEqual(result["objects"][1]["mask_index"], 0)
        self.assertEqual(result["objects"][1]["phrase"], "car")
        example = example_fixture()
        example["targets"][0]["instance_token"] = "truck-2"
        totals = Counter()
        score_prediction(result, example, totals)
        self.assertEqual(totals["true_positives"], 1)
        metadata = {
            key: {}
            for key in (
                "source_identity",
                "implementation_sha256",
                "extension_sha256",
                "tokens",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            save_prediction(
                Path(directory),
                {
                    "request_id": "q",
                    "sample_data_token": "f",
                    "scene_token": "s",
                    "point_count": 4,
                },
                result,
                "a" * 64,
                metadata,
            )
            file = next(Path(directory).glob("*.npz"))
            with np.load(file) as arrays:
                self.assertEqual(arrays["masks"].shape, (1, 4))
                self.assertEqual(
                    arrays["masks"][0].tolist(), [False, False, True, True]
                )

    def test_unicode_and_punctuation_preserved_around_multiple_markers(self):
        answer = "前方车辆，右侧行人。"
        objects = [
            {"status": "selected", "mask_index": 0},
            {"status": "selected", "mask_index": 1},
        ]
        output = render_segmented_answer(answer, [[0, 4], [5, 9]], objects)
        self.assertEqual(output, "前方车辆 <SEG>，右侧行人 <SEG>。")
        self.assertEqual(output.replace(" <SEG>", ""), answer)
        for record in objects:
            start, end = record["output_marker_span"]
            self.assertEqual(output[start:end], "<SEG>")

    def test_empty_selected_spatial_mask_cannot_create_output_marker(self):
        model = model_fixture().eval()
        instances = {"masks": torch.zeros(1, 4, dtype=torch.bool)}
        with (
            patch.object(
                model, "candidates", return_value=({}, instances, torch.zeros(1, 256))
            ),
            patch.object(
                model.head, "forward", return_value=torch.tensor([[20.0, -20.0]])
            ),
        ):
            result = model.segment_answer(
                make_prompt("truck"), torch.randn(3, 768), torch.zeros(4, 4), "truck"
            )
        self.assertEqual(result["objects"][0]["status"], "empty_spatial_mask")
        self.assertNotIn("<SEG>", result["answer"])

    def test_text_spans_reject_overlap_and_reserved_supervision(self):
        for answer, spans in (
            ("truck and car", [[0, 5], [4, 9]]),
            ("truck <SEG>", None),
            ("", None),
        ):
            with self.assertRaises(ValueError):
                answer_spans(answer, spans)


if __name__ == "__main__":
    unittest.main()
