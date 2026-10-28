"""Contrastive checks for the narrowly scoped text-only review rules."""
import importlib.util
import csv
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "nuscenes_category_analysis.py"
SPEC = importlib.util.spec_from_file_location("nuscenes_category_analysis", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class CategoryReviewTests(unittest.TestCase):
    def test_original_language_regressions(self):
        analysis.self_check()

    def test_scoped_negation_does_not_erase_existing_pedestrian(self):
        text = "A pedestrian is visible. No new dynamic obstacles or pedestrians detected."
        record = analysis.make_record("description", 1, text, "", "")
        self.assertEqual(record["states"]["pedestrian"], ["肯定提及", "否定新增"])
        self.assertFalse(record["review_reasons"])
        pedestrian = [m for m in record["mentions"] if m["category"] == "pedestrian"]
        for mention in pedestrian:
            start, end = mention["span"]
            self.assertEqual(text[start:end], mention["word"])

    def test_unqualified_or_separate_negative_is_not_rewritten(self):
        absent = analysis.make_record("description", 1, "A pedestrian is visible. No obstacles or pedestrians detected.", "", "")
        self.assertEqual(absent["states"]["pedestrian"], ["肯定提及", "否定存在"])
        self.assertTrue(any("同时有肯定与否定" in x for x in absent["review_reasons"]))
        independent = analysis.make_record("description", 1, "No new obstacles, but pedestrians are visible.", "", "")
        self.assertEqual(independent["states"]["pedestrian"], ["肯定提及"])

    def test_scoped_negation_in_hypothetical_is_not_asserted(self):
        for text in ("If no new obstacles or pedestrians are detected, the car can continue.",
                     "There might be no new obstacles or pedestrians."):
            record = analysis.make_record("description", 1, text, "", "")
            self.assertEqual(record["states"]["pedestrian"], ["条件或可能"], text)
            self.assertTrue(any("存在或类别含条件" in s for s in record["review_reasons"]))

    def test_frame_only_never_infers_category_from_question_or_answer(self):
        a = analysis.make_record("temporal_understanding", 1, "From frame 000 to frame 008.", "Where is the car?", "A car.")
        b = analysis.make_record("temporal_understanding", 1, "From frame 000 to frame 008.", "Where is the bus?", "A bus.")
        self.assertEqual(a["states"], b["states"])
        self.assertFalse(any(a["states"].values()))
        self.assertFalse(a["review_reasons"])
        self.assertEqual(analysis.review_flags(a), ["纯帧范围但目标未知"])
        self.assertEqual(analysis.primary_stratum(a), "纯帧范围但目标未知")

    def test_nearby_short_answers_remain_for_review(self):
        for text in ("Yes.", "From frame 000 to frame 008 the vehicle moves.", "From frame 000 to frame 008 a bus moves."):
            record = analysis.make_record("temporal_understanding", 1, text, "", "")
            self.assertFalse(record["auto_decision"], text)
        yes = analysis.make_record("temporal_understanding", 1, "Yes.", "", "")
        self.assertEqual(analysis.primary_stratum(yes), "无16类：是非短答")

    def test_sampling_groups_by_both_question_and_answer_without_losing_ids(self):
        records = [analysis.make_record("temporal_understanding", index, "Yes.", question, "")
                   for index, question in ((1, "Question A"), (2, "Question A"), (3, "Question B"))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = analysis.review_outputs(root, records, records)
            with (root / "人工抽查清单.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(summary["sample_groups"], 2)
            self.assertEqual(summary["sample_covered_records"], 3)
            self.assertEqual({r["原问题（仅辅助理解指代）"] for r in rows}, {"Question A", "Question B"})
            self.assertEqual({r["任务内序号"] for r in rows}, {"1；2", "3"})


if __name__ == "__main__":
    unittest.main()
