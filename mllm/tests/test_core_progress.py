import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from b4dl_core.io import digest
from scripts import inspect_core_progress as progress


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(root):
    dataset = root / "dataset"
    rows = [{"request_id": f"r{i}", "input_fingerprint": f"f{i}"} for i in range(2)]
    public_rows = []
    for split in ("internal_dev", "test"):
        split_rows = [dict(row, split=split) for row in rows]
        if split == "test":
            split_rows = [dict(row, request_id="test-" + row["request_id"]) for row in split_rows]
        public_rows.extend(split_rows)
        jsonl(dataset / (split + ".jsonl"), [{"request": row, "target": {}} for row in split_rows])
    jsonl(dataset / "requests.jsonl", public_rows)
    write(dataset / "manifest.json", {"dataset_id": "fixture", "files": {
        "requests.jsonl": progress.file_sha256(dataset / "requests.jsonl")}})
    request_path = root / "six-requests.jsonl"
    jsonl(request_path, rows)
    protocol = {"dataset_id": "fixture", "request_path": str(request_path),
                "files_sha256": {str(request_path): progress.file_sha256(request_path)}}
    protocol["protocol_id"] = digest(protocol)
    protocol_path = root / "protocol.json"
    write(protocol_path, protocol)
    evaluation = root / "evaluation-20261023-r1"
    write(evaluation / "status.json", {"status": "running", "pid": 123,
                                        "current_stage": "A3-internal_dev"})
    jsonl(evaluation / "A3-internal_dev" / "predictions.jsonl", [dict(rows[0], status="ok", text="Yes")])
    return dataset, protocol_path, evaluation, rows


class CoreProgressTests(unittest.TestCase):
    def test_dead_controller_does_not_inherit_running_and_inspection_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol, _, _ = fixture(root)
            before = {str(p): progress.file_sha256(p) for p in root.rglob("*") if p.is_file()}
            with patch.object(progress, "process_state", return_value={"state": "not_running"}):
                report = progress.inspect_run(root, dataset, protocol, check_processes=True)
            evaluation = report["evaluation"]
            self.assertEqual(evaluation["state"], "controller_missing")
            self.assertTrue(evaluation["stale_running_record"])
            self.assertEqual(evaluation["stages"]["A3-internal_dev"]["matched_requests"], 1)
            self.assertFalse(evaluation["quality_accepted"])
            self.assertEqual(before, {str(p): progress.file_sha256(p) for p in root.rglob("*") if p.is_file()})

    def test_copied_receipts_require_explicit_host_check(self):
        self.assertEqual(progress.process_state(1)["state"], "unverified")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol, _, _ = fixture(root)
            self.assertEqual(progress.inspect_run(root, dataset, protocol)["evaluation"]["state"], "unverified")

    def test_pid_reuse_command_birth_and_zombie_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            path = proc / "123"
            path.mkdir()
            (proc / "stat").write_text("btime 1000\n")
            fields = ["S"] + ["0"] * 18 + ["100"]
            (path / "stat").write_text("123 (python worker) " + " ".join(fields))
            (path / "cmdline").write_bytes(b"python\0train.py\0")
            options = dict(check=True, command=["python", "train.py"], proc_root=proc,
                           ticks=100, started="1970-01-01T00:16:41+00:00")
            self.assertEqual(progress.process_state(123, **options)["state"], "running")
            options["started"] = "1970-01-01T00:18:41+00:00"
            self.assertEqual(progress.process_state(123, **options)["reason"], "pid_start_mismatch")
            options["command"] = ["python", "other.py"]
            self.assertEqual(progress.process_state(123, **options)["reason"], "pid_command_mismatch")
            fields[0] = "Z"
            (path / "stat").write_text("123 (python worker) " + " ".join(fields))
            self.assertEqual(progress.process_state(123, **options)["reason"], "zombie_or_dead")
            self.assertEqual(progress.process_state(456, **options)["reason"], "pid_missing")

    def test_count_only_completion_and_duplicate_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol, evaluation, rows = fixture(root)
            stage = evaluation / "A3-internal_dev"
            run = {"dataset_id": "fixture", "split": "internal_dev"}
            write(stage / "run.json", run)
            write(stage / "completion.json", {"status": "inference_complete", "predictions": 2, "run": run})
            result = progress.inspect_run(root, dataset, protocol)["evaluation"]
            self.assertEqual(result["state"], "inconsistent")
            self.assertFalse(result["stages"]["A3-internal_dev"]["completion_verified"])
            jsonl(stage / "predictions.jsonl", [dict(rows[0], status="ok")] * 2)
            result = progress.inspect_run(root, dataset, protocol)["evaluation"]
            self.assertIn("duplicate_request:2", result["stages"]["A3-internal_dev"]["issues"])

    def test_stage_completion_requires_matching_dataset_and_prediction_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol, evaluation, rows = fixture(root)
            stage = evaluation / "A3-internal_dev"
            jsonl(stage / "predictions.jsonl", [dict(r, status="truncated") for r in rows])
            run = {"dataset_id": "fixture", "split": "internal_dev"}
            write(stage / "run.json", run)
            write(stage / "completion.json", {"status": "inference_complete", "predictions": 2, "run": run})
            result = progress.inspect_run(root, dataset, protocol)["evaluation"]
            self.assertTrue(result["stages"]["A3-internal_dev"]["completion_verified"])
            self.assertFalse(result["quality_accepted"])
            run["dataset_id"] = "another-dataset"
            write(stage / "run.json", run)
            write(stage / "completion.json", {"status": "inference_complete", "predictions": 2, "run": run})
            self.assertFalse(progress.inspect_run(root, dataset, protocol)["evaluation"]["stages"]["A3-internal_dev"]["completion_verified"])

    def test_torn_jsonl_or_changed_fingerprint_cannot_be_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text('{"request_id":"r0","input_fingerprint":"wrong","status":"ok"}\n{"request_id":')
            result = progress.prediction_progress(path, {"r0": "f0"})
            self.assertFalse(result["complete_coverage"])
            self.assertIn("request_identity_mismatch:1", result["issues"])
            self.assertIn("malformed_record:2", result["issues"])

    def test_sealed_completion_requires_prediction_receipt_and_journal_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol, evaluation, rows = fixture(root)
            stage = evaluation / "A3-internal_dev"
            jsonl(stage / "predictions.jsonl", [dict(r, status="ok", text="Yes") for r in rows])
            run = {"dataset_id": "fixture", "split": "internal_dev"}
            write(stage / "run.json", run)
            write(stage / "resume.json", {"schema": "b4dl-core-inference-resume-v1"})
            jsonl(stage / "input-receipts.jsonl", rows)
            jsonl(stage / "inference-journal.jsonl", rows)
            done = {"status": "inference_complete", "predictions": 2, "run": run}
            files = {"predictions_sha256": "predictions.jsonl", "input_receipts_sha256": "input-receipts.jsonl",
                     "journal_sha256": "inference-journal.jsonl"}
            for key, name in files.items(): done[key] = progress.file_sha256(stage / name)
            write(stage / "completion.json", done)
            def verified():
                return progress.inspect_run(root, dataset, protocol)["evaluation"]["stages"]["A3-internal_dev"]["completion_verified"]
            self.assertTrue(verified())
            for key, name in files.items():
                with self.subTest(file=name):
                    original = (stage / name).read_bytes()
                    (stage / name).write_bytes(original + b" ")
                    self.assertFalse(verified())
                    (stage / name).write_bytes(original)
                    write(stage / "completion.json", {k: v for k, v in done.items() if k != key})
                    self.assertFalse(verified())
                    write(stage / "completion.json", done)

    def test_frozen_request_and_protocol_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, path, _, _ = fixture(root)
            protocol = progress.read_json(path)
            protocol["dataset_id"] = "changed"
            write(path, protocol)
            with self.assertRaisesRegex(ValueError, "protocol identity"):
                progress.inspect_run(root, dataset, path)
            dataset, path, _, _ = fixture(root)
            (root / "six-requests.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "request file hash"):
                progress.inspect_run(root, dataset, path)
            dataset, path, _, _ = fixture(root)
            (dataset / "requests.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "Geometry request file hash"):
                progress.inspect_run(root, dataset, path)

    def test_training_failure_reports_saved_and_unsaved_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training/A3-seed42"
            write(training / "runtime.json", {"optimizer_steps": 3})
            write(training / "latest.json", {"step": 1})
            write(training / "error.json", {"message": "nonfinite training loss"})
            write(root / "jobs/A3-seed42/exit.json", {"return_code": 1})
            jsonl(training / "loss.jsonl", [{"step": 1}, {"step": 2}])
            result = progress.inspect_training(root, "A3-seed42")
            self.assertEqual(result["state"], "failed")
            self.assertEqual(result["completed_updates"], 2)
            self.assertEqual(result["unsaved_updates"], 1)
            self.assertFalse(result["completion_verified"])

    def test_training_completion_requires_exit_hash_checkpoint_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training/A3-seed42"
            identity = {"dataset_id": "fixture"}
            write(training / "runtime.json", {"optimizer_steps": 2, "identity": identity})
            checkpoint = training / "step-2.pt"
            checkpoint.write_bytes(b"test-checkpoint")
            write(training / "latest.json", {"step": 2, "identity": identity, "checkpoint": checkpoint.name,
                                              "checkpoint_sha256": progress.file_sha256(checkpoint)})
            jsonl(training / "loss.jsonl", [{"step": 1}, {"step": 2}])
            completion = training / "completion.json"
            write(completion, {"status": "training_complete", "optimizer_steps": 2, "identity": identity})
            write(root / "jobs/A3-seed42/exit.json", {"return_code": 0,
                                                    "completion_sha256": progress.file_sha256(completion)})
            self.assertTrue(progress.inspect_training(root, "A3-seed42")["completion_verified"])
            checkpoint.write_bytes(b"changed")
            result = progress.inspect_training(root, "A3-seed42")
            self.assertFalse(result["completion_verified"])
            self.assertEqual(result["state"], "inconsistent")


if __name__ == "__main__":
    unittest.main()
