"""Full-denominator evaluation and deterministic, prediction-independent controls."""
from collections import Counter, defaultdict, deque
import numpy as np

from .io import digest, unique_index
from .rewards import score_response


def mean(values):
    return float(np.mean(values)) if values else None


def scene_interval(rows, value="correct", repetitions=2000, seed=42):
    """Cluster bootstrap: resample scenes, retaining every row in each scene."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_id"]].append(float(row[value]))
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if len(groups) < 2:
        return {"low": None, "high": None, "scenes": len(groups), "reason": "fewer_than_two_scenes"}
    keys = sorted(groups)
    sums = np.array([sum(groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sample = rng.integers(len(keys), size=len(keys))
        values.append(float(sums[sample].sum()/counts[sample].sum()))
    low, high = np.quantile(values, [.025, .975])
    return {"low": float(low), "high": float(high), "scenes": len(keys),
            "repetitions": repetitions, "seed": seed, "unit": "scene", "estimand": "question_micro_mean"}


def aggregate(rows):
    by_label = defaultdict(list)
    errors = defaultdict(list)
    for row in rows:
        by_label[row["truth"]].append(row["correct"])
        for key, value in row["absolute_errors"].items():
            errors[key].append(value)
    return {"n": len(rows), "accuracy": mean([r["correct"] for r in rows]),
            "balanced_accuracy_observed_labels": mean([mean(v) for v in by_label.values()]),
            "label_counts": dict(Counter(r["truth"] for r in rows)),
            "answer_valid_rate": mean([r["answer_valid"] for r in rows]),
            "evidence_valid_rate": mean([r["evidence_valid"] for r in rows]),
            "evidence_consistent_rate": mean([r["evidence_consistent"] for r in rows]),
            "mean_evidence_score": mean([r["evidence_score"] for r in rows]),
            "mean_combined_reward": mean([r["combined_reward"] for r in rows]),
            "numeric_errors_valid_only": {k: {"n": len(v), "mae": mean(v)} for k, v in sorted(errors.items())},
            "prediction_distribution": dict(Counter(r["answer"] or "INVALID" for r in rows)),
            "status_counts": dict(Counter(r["status"] for r in rows))}


def opposite_pairs(requests, targets):
    """Maximum disjoint bipartite matching per exact prompt and frame length."""
    buckets = defaultdict(list)
    for r in requests:
        if r["kind"] == "binary":
            buckets[(r["split"], r["task"], r["predicate"], r["prompt"], r["feature_length"])].append(r)
    pairs = []
    for key, rows in sorted(buckets.items()):
        left = sorted((r for r in rows if targets[r["request_id"]]["answer"] == "Yes"), key=lambda r: r["request_id"])
        right = sorted((r for r in rows if targets[r["request_id"]]["answer"] == "No"), key=lambda r: r["request_id"])
        edges = [[j for j, b in enumerate(right) if a["scene_id"] != b["scene_id"]] for a in left]
        match_left, match_right = {}, {}
        for start in range(len(left)):
            queue, visited_left, parent_right = deque([start]), {start}, {}
            end = None
            while queue and end is None:
                i = queue.popleft()
                for j in edges[i]:
                    if j in parent_right:
                        continue
                    parent_right[j] = i
                    if j not in match_right:
                        end = j
                        break
                    nxt = match_right[j]
                    if nxt not in visited_left:
                        visited_left.add(nxt)
                        queue.append(nxt)
            while end is not None:
                i = parent_right[end]
                previous = match_left.get(i)
                match_left[i], match_right[end] = end, i
                end = previous
        for i, j in sorted(match_left.items()):
            a, b = left[i], right[j]
            pairs.append({"pair_id": digest([a["request_id"], b["request_id"]])[:24],
                          "yes_id": a["request_id"], "no_id": b["request_id"],
                          "task": key[1], "prompt_sha256": digest(key[3]), "feature_length": key[4]})
    return pairs


def evaluate(dataset, predictions, run, repetitions=2000):
    required = {"dataset_id", "split", "condition", "model_id", "checkpoint_sha256", "prediction_kind"}
    if set(run) != required or run["dataset_id"] != dataset.manifest["dataset_id"]:
        raise ValueError("run identity must match the dataset and required fields")
    if run["prediction_kind"] not in ("model", "fixture", "prior") or not run["model_id"]:
        raise ValueError("explicit model/prior/fixture identity required")
    if run["prediction_kind"] == "model" and (not isinstance(run["checkpoint_sha256"], str) or len(run["checkpoint_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in run["checkpoint_sha256"])):
        raise ValueError("model evaluation requires checkpoint manifest SHA256")
    if run["condition"] not in ("identity", "zero", "repeat_first", "shuffle_features", "cross_scene"):
        raise ValueError("unknown condition")
    if not dataset.requests or {r["split"] for r in dataset.requests} != {run["split"]}:
        raise ValueError("evaluate exactly one nonempty split")
    indexed = unique_index(predictions, "request_id")
    if set(indexed) - {r["request_id"] for r in dataset.requests}:
        raise ValueError("unknown prediction IDs")
    rows = []
    for request in dataset.requests:
        ident = request["request_id"]
        pred = indexed.get(ident)
        if pred is not None:
            if set(pred) != {"request_id", "input_fingerprint", "status", "text"} or pred["input_fingerprint"] != request["input_fingerprint"]:
                raise ValueError("prediction input identity mismatch")
            if pred["status"] not in ("ok", "error", "unavailable", "truncated"):
                raise ValueError("unknown prediction status")
        status = pred["status"] if pred is not None else "missing"
        target = dataset.targets[ident]
        score = score_response(request, target, pred["text"] if pred else "", status)
        rows.append({"request_id": ident, "scene_id": request["scene_id"], "task": request["task"],
                     "predicate": request["predicate"], "truth": target["answer"], "status": status, **score})
    index = unique_index(rows, "request_id")
    groups, tasks = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row["task"] + "/" + row["predicate"]].append(row)
        tasks[row["task"]].append(row)
    pairs = opposite_pairs(dataset.requests, dataset.targets)
    pair_scores = [{**p, "both_correct": index[p["yes_id"]]["correct"] and index[p["no_id"]]["correct"],
                    "answer_changes": index[p["yes_id"]]["answer_valid"] and index[p["no_id"]]["answer_valid"]
                    and index[p["yes_id"]]["answer"] != index[p["no_id"]]["answer"]} for p in pairs]
    orbits = defaultdict(list)
    for request in dataset.requests:
        if request["kind"] == "mcq":
            orbits[request["semantic_id"]].append(request)
    orbit_scores = []
    for ident, requests in sorted(orbits.items()):
        if len(requests) != 4 or {r["option_rotation"] for r in requests} != set(range(4)):
            raise ValueError("incomplete option rotation group")
        answers = [r["options"].get(index[r["request_id"]]["answer"]) for r in requests]
        orbit_scores.append({"semantic_id": ident, "semantic_consistent": None not in answers and len(set(answers)) == 1,
                             "all_correct": all(index[r["request_id"]]["correct"] for r in requests)})
    return {"schema": "b4dl-core-evaluation-v1", "run": run, "summary": aggregate(rows),
            "scene_bootstrap_accuracy": scene_interval(rows, repetitions=repetitions),
            "by_task": {k: aggregate(v) for k, v in sorted(tasks.items())},
            "by_task_predicate": {k: aggregate(v) for k, v in sorted(groups.items())},
            "opposite_pairs": {"n": len(pairs), "both_correct_rate": mean([p["both_correct"] for p in pair_scores]),
                               "answer_change_rate": mean([p["answer_changes"] for p in pair_scores]), "rows": pair_scores},
            "option_rotations": {"n": len(orbit_scores), "semantic_consistency": mean([o["semantic_consistent"] for o in orbit_scores]),
                                 "all_correct_rate": mean([o["all_correct"] for o in orbit_scores]), "rows": orbit_scores}, "rows": rows}


def compare_reports(base, candidate, repetitions=2000):
    for key in ("dataset_id", "split"):
        if base["run"][key] != candidate["run"][key]:
            raise ValueError("paired comparison requires identical dataset and split")
    left, right = (unique_index(r["rows"], "request_id") for r in (base, candidate))
    if set(left) != set(right) or not left:
        raise ValueError("paired comparison requires same nonempty request denominator")
    rows = []
    for ident, a in left.items():
        b = right[ident]
        if (a["scene_id"], a["truth"]) != (b["scene_id"], b["truth"]):
            raise ValueError("paired truth mismatch")
        rows.append({"request_id": ident, "scene_id": a["scene_id"],
                     "delta": int(b["correct"]) - int(a["correct"]),
                     "repair": not a["correct"] and b["correct"], "regression": a["correct"] and not b["correct"]})
    return {"base_run": base["run"], "candidate_run": candidate["run"], "n": len(rows),
            "accuracy_delta": mean([r["delta"] for r in rows]),
            "repairs": sum(r["repair"] for r in rows), "regressions": sum(r["regression"] for r in rows),
            "both_correct": sum(left[r["request_id"]]["correct"] and right[r["request_id"]]["correct"] for r in rows),
            "both_wrong": sum(not left[r["request_id"]]["correct"] and not right[r["request_id"]]["correct"] for r in rows),
            "scene_bootstrap_delta": scene_interval(rows, "delta", repetitions), "rows": rows}


def prior_predictions(train, evaluation, policy="train_majority"):
    """Answer-only diagnostics: evidence stays invalid, no held-out label fitting."""
    import json
    if policy not in ("train_majority", "always_yes", "always_no", "always_a"):
        raise ValueError("unknown prior")
    if not train.requests or {r["split"] for r in train.requests} != {"train"}:
        raise ValueError("prior fitting requires train only")
    if train.manifest["dataset_id"] != evaluation.manifest["dataset_id"]:
        raise ValueError("prior training dataset mismatch")
    counts = defaultdict(Counter)
    for r in train.requests:
        counts[(r["task"], r["predicate"])][train.targets[r["request_id"]]["answer"]] += 1
    result = []
    for r in evaluation.requests:
        labels = sorted(r["options"]) if r["kind"] == "mcq" else ["No", "Yes"]
        votes = counts[(r["task"], r["predicate"])]
        answer = min(labels, key=lambda k: (-votes[k], k))
        if policy != "train_majority":
            answer = {"always_yes": "Yes", "always_no": "No", "always_a": "A"}[policy]
            if answer not in labels:
                answer = labels[0]
        result.append({"request_id": r["request_id"], "input_fingerprint": r["input_fingerprint"],
                       "status": "ok", "text": json.dumps({"answer": answer, "evidence": {}})})
    return result
