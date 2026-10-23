"""Deterministic offline scoring; this module does not implement RL training."""
import json
import math

from .geometry import motion_truth, sector


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def parse_response(text, request):
    """Exact JSON contract: no prose salvage, duplicate keys, bools or NaN."""
    failure = {"answer": None, "evidence": {}, "answer_valid": False, "evidence_valid": False}
    try:
        if not isinstance(text, str):
            return failure
        response = json.loads(text, object_pairs_hook=_pairs, parse_constant=_bad_constant)
        if not isinstance(response, dict) or set(response) != {"answer", "evidence"}:
            return failure
        allowed = set(request["options"]) if request["kind"] == "mcq" else {"Yes", "No"}
        answer = response["answer"]
        if not isinstance(answer, str) or answer not in allowed:
            return failure
        ev = response["evidence"]
        valid = isinstance(ev, dict) and set(ev) == set(request["evidence_keys"])
        if valid:
            valid = all(type(v) in (int, float) and math.isfinite(v) for v in ev.values())
        if valid:
            valid = all(v >= 0 for k, v in ev.items() if k.startswith("distance_") or k in ("d0", "d1", "count"))
        if valid and "count" in ev:
            valid = type(ev["count"]) is int
        return {"answer": answer, "evidence": ev if valid else {}, "answer_valid": True, "evidence_valid": bool(valid)}
    except (ValueError, TypeError, OverflowError, RecursionError):
        return failure


def evidence_answer(request, evidence):
    task = request["task"]
    if task == "existence":
        return "Yes" if evidence["count"] > 0 else "No"
    if task == "position":
        return "Yes" if sector(evidence["x"], evidence["y"]) == request["predicate"] else "No"
    if task == "motion":
        if abs(evidence["d1"] - evidence["d0"] - evidence["delta_d"]) > .11:
            return None
        return "Yes" if motion_truth(evidence)[request["predicate"]] else "No"
    if task == "distance":
        distances = {k: evidence["distance_" + v] for k, v in request["options"].items()}
        best = min(distances.values())
        labels = [k for k, v in distances.items() if abs(v-best) <= 1e-6]
        return labels[0] if len(labels) == 1 else None
    raise ValueError("unknown task")


def score_response(request, target, text, status="ok", numeric_tolerance_m=1.):
    if not 0 < numeric_tolerance_m < float("inf"):
        raise ValueError("tolerance must be finite and positive")
    if request["request_id"] != target["request_id"] or set(target["evidence"]) != set(request["evidence_keys"]):
        raise ValueError("target does not match request")
    truth = parse_response(json.dumps({"answer": target["answer"], "evidence": target["evidence"]}), request)
    if not truth["evidence_valid"] or evidence_answer(request, truth["evidence"]) != target["answer"]:
        raise ValueError("invalid or contradictory target")
    parsed = parse_response(text if status == "ok" else "", request)
    correct = parsed["answer_valid"] and parsed["answer"] == target["answer"]
    errors, components, consistent = {}, {}, False
    if parsed["evidence_valid"]:
        for key, truth in target["evidence"].items():
            error = abs(parsed["evidence"][key] - truth)
            errors[key] = error
            components[key] = float(error == 0) if key == "count" else max(0., 1. - error/numeric_tolerance_m)
        try:
            consistent = evidence_answer(request, parsed["evidence"]) == parsed["answer"]
        except ValueError:
            consistent = False
    # A coherent fabricated chain cannot score without numerical agreement to GT.
    evidence_score = sum(components.values()) / len(components) if components and consistent else 0.
    return {"answer": parsed["answer"], "correct": bool(correct),
            "answer_valid": parsed["answer_valid"], "evidence_valid": parsed["evidence_valid"],
            "evidence_consistent": bool(consistent), "absolute_errors": errors,
            "evidence_score": evidence_score, "answer_reward": float(correct),
            "combined_reward": .7 * float(correct) + .3 * evidence_score}
