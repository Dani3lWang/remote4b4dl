import copy
import json
import unittest

import numpy as np

from b4dl_core.reasoning.data import public_request, verify_truth
from b4dl_core.reasoning.protocol import InvalidTrace, encode_trace, make_trace, parse_trace
from b4dl_core.reasoning.rewards import score_trace, validate_target


def example(task):
    common = {"request_id": "r", "scene_id": "s", "split": "train", "frames": [0],
              "feature_length": 9, "options": {}, "kind": "binary", "predicate": "",
              "input_fingerprint": "parent", "question": "fixture"}
    if task == "existence":
        request = {**common, "task": task, "question": "Is there a car supported by LiDAR returns?"}
        target = {"request_id": "r", "answer": "Yes", "evidence": {"count": 1},
                  "provenance": {"annotations": ["a0"]}}
        endpoints = None
    elif task == "position":
        request = {**common, "task": task, "predicate": "front", "question": "Is the nearest LiDAR-supported car in the front sector?"}
        target = {"request_id": "r", "answer": "Yes", "evidence": {"x": 10., "y": 0.},
                  "provenance": {"instances": ["a"]}}
        endpoints = None
    elif task == "motion":
        request = {**common, "task": task, "frames": [0, 8], "predicate": "closer",
                   "question": "Follow the same LiDAR-supported car that is nearest."}
        target = {"request_id": "r", "answer": "Yes", "evidence": {"d0": 10., "d1": 7., "delta_d": -3., "delta_y": 1.},
                  "provenance": {"instances": ["a"], "annotations": ["a0", "a8"]}}
        endpoints = {"y0": 0., "y1": 1.}
    else:
        options = {"A": "car", "B": "truck", "C": "bus", "D": "bicycle"}
        request = {**common, "task": "distance", "kind": "mcq", "options": options}
        target = {"request_id": "r", "answer": "A", "evidence": {"distance_car": 2., "distance_truck": 4., "distance_bus": 6., "distance_bicycle": 8.},
                  "provenance": {"instances": ["a", "b", "c", "d"]}}
        endpoints = None
    target["trace"] = make_trace(request, target, endpoints)
    return request, target


class ReasoningProtocolTests(unittest.TestCase):
    def test_teacher_traces_have_grounded_steps(self):
        for task in ("existence", "position", "motion", "distance"):
            with self.subTest(task=task):
                request, target = example(task)
                text = encode_trace(target["trace"])
                parsed = parse_trace(text, request)
                score = score_trace(request, target, text)
                self.assertEqual(score["step_scores"], [1.]*len(parsed["steps"]))
                self.assertTrue(score["correct"])
                self.assertEqual(len(parsed["step_char_ends"]), len(parsed["steps"]))
                self.assertTrue(all(text[n-1] == "}" for n in parsed["step_char_ends"]))

    def test_correct_answer_does_not_hide_wrong_or_fabricated_process(self):
        request, target = example("motion")
        fake = copy.deepcopy(target["trace"])
        fake["steps"][0]["value"].update(d0=100., d1=97., y0=10., y1=11.)
        score = score_trace(request, target, encode_trace(fake))
        self.assertTrue(score["correct"])
        self.assertEqual(score["step_scores"], [0., 0., 0.])
        fake = copy.deepcopy(target["trace"])
        fake["steps"][1]["value"]["delta_y"] = 0.
        score = score_trace(request, target, encode_trace(fake))
        self.assertEqual(score["step_scores"][0], 1.)
        self.assertEqual(score["step_scores"][1:], [0., 0.])

    def test_tolerance_and_dependency_cap(self):
        request, target = example("position")
        candidate = copy.deepcopy(target["trace"])
        candidate["steps"][0]["value"]["x"] += .25
        score = score_trace(request, target, encode_trace(candidate))
        self.assertEqual(score["step_scores"], [.75, .75, .75])
        candidate["steps"][1]["value"] = "left"
        score = score_trace(request, target, encode_trace(candidate))
        self.assertEqual(score["step_scores"], [.75, 0., 0.])

    def test_invalid_formats_are_penalized_without_salvage(self):
        request, target = example("existence")
        correct = encode_trace(target["trace"])
        candidates = [correct[:-1], correct+" prose", correct.replace('"count":1', '"count":true'),
                      correct.replace('"count":1', '"count":NaN'),
                      correct.replace('"count":1', '"count":1,"count":1'),
                      correct.replace('"depends_on":[0]', '"depends_on":[1]'),
                      correct.replace('"answer":"Yes"', '"answer":"Maybe"')]
        for text in candidates:
            with self.subTest(text=text):
                self.assertEqual(score_trace(request, target, text)["answer_reward"], -1.)
        self.assertEqual(score_trace(request, target, correct, "truncated")["answer_reward"], -1.)
        extra = copy.deepcopy(target["trace"])
        extra["steps"].append(extra["steps"][-1])
        with self.assertRaises(InvalidTrace): parse_trace(encode_trace(extra), request)

    def test_contradictory_truth_fails_closed(self):
        request, target = example("motion")
        target["trace"]["steps"][1]["value"]["delta_d"] = 9.
        with self.assertRaises(ValueError): score_trace(request, target, "")
        target = example("motion")[1]
        target["request_id"] = "other"
        with self.assertRaises(ValueError): validate_target(request, target)

    def test_public_request_contains_no_truth(self):
        request, target = example("motion")
        public = public_request(request)
        self.assertNotIn("answer", public)
        self.assertNotIn("provenance", public)
        self.assertNotIn("evidence", public)
        self.assertNotIn("a8", json.dumps(public))
        self.assertNotIn(str(target["evidence"]["d0"]), public["prompt"])

    def test_instance_selection_and_crop_support(self):
        request, target = example("motion")
        frames = {("s", 0): {"objects": [{"class": "car", "instance": "a", "annotation": "a0",
                                           "distance": 10., "xyz": np.array([10., 0., 0.])}], "unsupported": set()},
                  ("s", 8): {"objects": [{"class": "car", "instance": "a", "annotation": "a8",
                                           "distance": 7., "xyz": np.array([np.sqrt(48), 1., 0.])},
                                          {"class": "car", "instance": "b", "annotation": "b8",
                                           "distance": 1., "xyz": np.array([1., 0., 0.])}], "unsupported": set()}}
        self.assertEqual(verify_truth(request, target, frames), {"y0": 0., "y1": 1.})
        frames[("s", 8)]["objects"] = frames[("s", 8)]["objects"][1:]
        with self.assertRaises(ValueError): verify_truth(request, target, frames)
        request, target = example("existence")
        target.update(answer="No", evidence={"count": 0}, provenance={"annotations": []})
        frames[("s", 0)] = {"objects": [], "unsupported": {"car"}}
        with self.assertRaises(ValueError): verify_truth(request, target, frames)


if __name__ == "__main__":
    unittest.main()
