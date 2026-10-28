"""Read-only core experiment status; no model imports, GPU work or job launch."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else None


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def process_state(pid, *, check=False, command=None, script=None, started=None,
                  proc_root=Path("/proc"), ticks=None):
    """Verify command and process birth, not just a potentially reused PID.

    Explicit opt-in is required: copied remote receipts must not query local PIDs.
    """
    if not check:
        return {"state": "unverified", "reason": "process_check_not_requested"}
    if not proc_root.is_dir():
        return {"state": "unverified", "reason": "proc_unavailable"}
    if not isinstance(pid, int) or pid < 1:
        return {"state": "unverified", "reason": "missing_pid"}
    path = proc_root / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return {"state": "not_running", "reason": "zombie_or_dead", "pid": pid}
        actual = (path / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        matches = actual == command if command else bool(script and str(script) in actual)
        if not matches:
            return {"state": "not_running", "reason": "pid_command_mismatch", "pid": pid}
        if not started:
            return {"state": "unverified", "reason": "missing_process_start", "pid": pid}
        boot = next(int(line.split()[1]) for line in (proc_root / "stat").read_text().splitlines()
                    if line.startswith("btime "))
        frequency = ticks if ticks is not None else os.sysconf("SC_CLK_TCK")
        born = boot + int(fields[19]) / frequency
        recorded = datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
        if abs(recorded - born) > 5:
            return {"state": "not_running", "reason": "pid_start_mismatch", "pid": pid}
        return {"state": "running", "pid": pid}
    except FileNotFoundError:
        return {"state": "not_running", "reason": "pid_missing", "pid": pid}
    except (OSError, ValueError, IndexError, StopIteration, AttributeError) as error:
        return {"state": "unverified", "reason": type(error).__name__, "pid": pid}


def expected_requests(path, split=None):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Request denominator unavailable: {path}")
    result = {}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            if split is not None and row.get("split") != split:
                continue
            ident = row["request_id"]
            if ident in result:
                raise ValueError(f"Duplicate expected request ID: {ident}")
            result[ident] = row["input_fingerprint"]
    return result


def prediction_progress(path, expected):
    path = Path(path)
    result = {"requested": len(expected), "rows": 0, "matched_requests": 0,
              "status_counts": {}, "issues": [], "complete_coverage": False}
    if not path.exists():
        return result
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    result["sha256"] = hashlib.sha256(content).hexdigest()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        result["issues"].append("file_changed_during_read")
    if content and not content.endswith(b"\n"):
        result["issues"].append("unterminated_last_record")
    seen, matched, counts = set(), set(), Counter()
    for number, line in enumerate(content.splitlines(), 1):
        try:
            row = json.loads(line)
            ident = row["request_id"]
            result["rows"] += 1
            counts[row["status"]] += 1
            if ident in seen:
                result["issues"].append(f"duplicate_request:{number}")
            seen.add(ident)
            if ident not in expected or row.get("input_fingerprint") != expected[ident]:
                result["issues"].append(f"request_identity_mismatch:{number}")
            else:
                matched.add(ident)
            if row["status"] not in {"ok", "truncated", "unavailable", "error"}:
                result["issues"].append(f"unknown_status:{number}")
        except (ValueError, KeyError, TypeError):
            result["issues"].append(f"malformed_record:{number}")
    result.update(matched_requests=len(matched), status_counts=dict(counts))
    result["complete_coverage"] = bool(expected) and len(matched) == len(expected) and not result["issues"]
    return result


def inspect_training(root, job_name, check_processes=False):
    job, training = root / "jobs" / job_name, root / "training" / job_name
    process = read_json(job / "process.json") or {}
    exited = read_json(job / "exit.json")
    runtime = read_json(training / "runtime.json") or {}
    latest = read_json(training / "latest.json") or {}
    completed = read_json(training / "completion.json")
    error = read_json(training / "error.json")
    live = process_state(process.get("pid"), check=check_processes, command=process.get("command"),
                         started=process.get("time_utc"))
    last, rows, issues = {}, 0, []
    loss = training / "loss.jsonl"
    if loss.exists():
        with loss.open(encoding="utf-8-sig") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if last and row["step"] != last["step"] + 1:
                        issues.append("nonconsecutive_loss_steps")
                    last, rows = row, rows + 1
                except (ValueError, KeyError):
                    issues.append("malformed_loss_record")
    budget, step = runtime.get("optimizer_steps"), last.get("step", 0)
    checkpoint = training / Path(latest.get("checkpoint", "missing")).name
    valid_completion = bool(
        completed and completed.get("status") == "training_complete"
        and budget and completed.get("optimizer_steps") == budget and step == budget
        and rows == budget - completed.get("resumed_from_step", 0)
        and latest.get("step") == budget and completed.get("identity") == latest.get("identity")
        and latest.get("identity") == runtime.get("identity")
        and checkpoint.is_file() and file_sha256(checkpoint) == latest.get("checkpoint_sha256")
        and exited and exited.get("return_code") == 0
        and exited.get("completion_sha256") == file_sha256(training / "completion.json")
        and not error and not issues
    )
    if completed and not valid_completion:
        issues.append("completion_evidence_disagrees")
    if valid_completion:
        state = "completed"
    elif error or (exited and exited.get("return_code") != 0):
        state = "failed"
    elif issues:
        state = "inconsistent"
    elif live["state"] == "running":
        state = "running"
    elif process and live["state"] == "not_running":
        state = "interrupted"
    else:
        state = "unverified" if process else "not_started"
    return {"state": state, "process": live, "completed_updates": step,
            "planned_updates": budget, "logged_updates": rows,
            "completion_fraction": step / budget if budget else None,
            "last_loss": last, "latest_checkpoint": latest,
            "unsaved_updates": max(0, step - latest.get("step", 0)),
            "completion_verified": valid_completion, "error": error, "exit": exited,
            "issues": issues, "quality_evaluation_completed": False}


def inspect_evaluation(directory, dataset, protocol, check_processes=False):
    status = read_json(directory / "status.json") or {}
    controller = process_state(status.get("pid"), check=check_processes,
                               script=directory / "run_evaluation.py", started=status.get("started_utc"))
    # Match GeometryDataset: split files wrap request/target; requests.jsonl is
    # the canonical public-input ledger used by actual model inference.
    expected = {split: expected_requests(dataset / "requests.jsonl", split) for split in ("internal_dev", "test")}
    expected["six-task"] = expected_requests(Path(protocol["request_path"]))
    stages = {}
    for split, rows in expected.items():
        for arm in ("A0", "A3"):
            name = arm + "-" + split
            stage = directory / name
            progress = prediction_progress(stage / "predictions.jsonl", rows)
            done = read_json(stage / "completion.json")
            run = read_json(stage / "run.json")
            marker_status = "six_task_inference_complete" if split == "six-task" else "inference_complete"
            count_key = "requests" if split == "six-task" else "predictions"
            identity_matches = bool(run) and (
                run.get("protocol_id") == protocol.get("protocol_id")
                and run.get("identity", {}).get("dataset_id") == protocol.get("dataset_id")
                and run.get("arm") == arm
                if split == "six-task" else
                bool(run) and run.get("dataset_id") == protocol.get("dataset_id") and run.get("split") == split
            )
            progress["completion_verified"] = bool(
                done and run and done.get("status") == marker_status
                and done.get(count_key) == len(rows) and done.get("run") == run
                and identity_matches and progress["complete_coverage"]
            )
            # New runs seal exports and the joint geometry journal. Keep old
            # receipts readable, but never ignore a supplied completion hash.
            sealed = (stage / "resume.json").exists() or bool(run and run.get("schema") == "b4dl-core-six-task-inference-v2")
            if done:
                for key, filename in (("predictions_sha256", "predictions.jsonl"),
                                      ("input_receipts_sha256", "input-receipts.jsonl"),
                                      ("journal_sha256", "inference-journal.jsonl")):
                    required = sealed and (split != "six-task" or key == "predictions_sha256")
                    if key in done or required:
                        path = stage / filename
                        if not path.is_file() or done.get(key) != file_sha256(path):
                            progress["completion_verified"] = False
                            progress["issues"].append("completion_file_hash_mismatch:" + filename)
            if done and not progress["completion_verified"]:
                progress["issues"].append("completion_evidence_disagrees")
            stages[name] = progress
    recorded = status.get("status")
    # Even complete predictions require the metric/comparison stages' receipts.
    required_reports = [directory / f"{split}-comparison.json" for split in ("internal_dev", "test")]
    required_reports += [directory / f"{arm}-six-task-metrics" / "completion.json" for arm in ("A0", "A3")]
    all_complete = all(p["completion_verified"] for p in stages.values())
    reports_present = all(read_json(p) for p in required_reports)
    if any(p["issues"] for p in stages.values()):
        state = "inconsistent"
    elif recorded == "evaluation_complete":
        state = "completion_receipts_present" if all_complete and reports_present else "inconsistent"
    elif recorded == "evaluation_failed":
        state = "failed"
    elif controller["state"] == "running":
        state = "running"
    elif recorded == "running" and controller["state"] == "not_running":
        state = "controller_missing"
    else:
        state = "unverified" if status else "not_started"
    return {"state": state, "recorded_status": recorded, "controller": controller,
            "stale_running_record": recorded == "running" and controller["state"] == "not_running",
            "current_stage": status.get("current_stage"), "stages": stages,
            "quality_accepted": False,
            "note": "Controller absence does not prove all child workers exited; no partial-prefix quality score is computed."}


def inspect_run(root, dataset, protocol_path, job_name="A3-seed42",
                evaluation_name="evaluation-20261023-r1", check_processes=False):
    protocol = read_json(protocol_path)
    if not protocol or not protocol.get("request_path"):
        raise ValueError("A six-task protocol with request_path is required")
    identity = {key: value for key, value in protocol.items() if key != "protocol_id"}
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != protocol.get("protocol_id"):
        raise ValueError("Six-task protocol identity mismatch")
    if file_sha256(protocol["request_path"]) != protocol.get("files_sha256", {}).get(protocol["request_path"]):
        raise ValueError("Six-task request file hash mismatch")
    manifest = read_json(dataset / "manifest.json")
    if not manifest or manifest.get("dataset_id") != protocol.get("dataset_id"):
        raise ValueError("Dataset/protocol identity mismatch")
    if file_sha256(dataset / "requests.jsonl") != manifest.get("files", {}).get("requests.jsonl"):
        raise ValueError("Geometry request file hash mismatch")
    return {"schema": "b4dl-core-progress-v1", "captured_utc": datetime.now(timezone.utc).isoformat(),
            "run_root": str(root), "request_sources": {"dataset": str(dataset), "protocol": str(protocol_path)},
            "training": inspect_training(root, job_name, check_processes),
            "evaluation": inspect_evaluation(root / evaluation_name, dataset, protocol, check_processes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "dataset", "six-task-protocol"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--job-name", default="A3-seed42")
    parser.add_argument("--evaluation-name", default="evaluation-20261023-r1")
    parser.add_argument("--check-local-processes", action="store_true",
                        help="Only on the owning Linux host; copied receipts cannot prove process liveness.")
    parser.add_argument("--output", type=Path, help="Write a new snapshot; never overwrite an existing file.")
    args = parser.parse_args()
    report = inspect_run(args.run_root, args.dataset, args.six_task_protocol, args.job_name,
                         args.evaluation_name, args.check_local_processes)
    text = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
