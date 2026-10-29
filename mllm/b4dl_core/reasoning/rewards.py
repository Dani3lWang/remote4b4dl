"""Deterministic process rewards grounded in verified, private geometry."""
import math

from ..geometry import motion_truth, sector
from .protocol import InvalidTrace, encode_trace, parse_trace, template


def _numeric_score(actual, truth, tolerance):
    errors = {k: abs(actual[k] - truth[k]) for k in truth}
    scores = [float(error == 0) if key == "count" else max(0., 1. - error/tolerance)
              for key, error in errors.items()]
    return min(scores), errors


def _derived_consistent(request, steps, index, arithmetic_tolerance):
    step, prev = steps[index], steps[index-1]["value"]
    value, op = step["value"], step["op"]
    if op == "exists":
        return value == (prev["count"] > 0)
    if op == "sector":
        try:
            return value == sector(prev["x"], prev["y"])
        except ValueError:
            return False
    if op == "predicate":
        if request["task"] == "position":
            return value == (prev == request["predicate"])
        return value == motion_truth(prev)[request["predicate"]]
    if op == "delta":
        return (abs(value["delta_d"] - (prev["d1"]-prev["d0"])) <= arithmetic_tolerance
                and abs(value["delta_y"] - (prev["y1"]-prev["y0"])) <= arithmetic_tolerance)
    if op == "argmin":
        distances = {label: prev["distance_" + name] for label, name in request["options"].items()}
        best = min(distances.values())
        winners = [label for label, distance in distances.items() if abs(distance-best) <= 1e-6]
        return len(winners) == 1 and value == winners[0]
    raise ValueError("unsupported derived operation")


def validate_target(request, target):
    if request["request_id"] != target["request_id"]:
        raise ValueError("request/target identity mismatch")
    truth = parse_trace(encode_trace(target["trace"]), request)
    if truth["answer"] != target["answer"]:
        raise ValueError("teacher answer mismatch")
    for i in range(1, len(truth["steps"])):
        if not _derived_consistent(request, truth["steps"], i, 1e-8):
            raise ValueError("contradictory teacher step")
    last = truth["steps"][-1]["value"]
    derived_answer = last if request["task"] == "distance" else ("Yes" if last else "No")
    if derived_answer != truth["answer"]:
        raise ValueError("teacher steps disagree with answer")
    return truth


def score_trace(request, target, text, status="ok", numeric_tolerance_m=1., arithmetic_tolerance_m=.01):
    """Returns scores in [0,1], or -1 outcome for a malformed completion.

    Broken private truth is an exception, never a model failure reward.
    """
    if not (math.isfinite(numeric_tolerance_m) and numeric_tolerance_m > 0
            and math.isfinite(arithmetic_tolerance_m) and arithmetic_tolerance_m > 0):
        raise ValueError("invalid reward tolerance")
    truth = validate_target(request, target)
    count = len(template(request))
    try:
        if status != "ok":
            raise InvalidTrace("completion did not finish")
        parsed = parse_trace(text, request)
    except InvalidTrace as error:
        return {"valid": False, "answer": None, "correct": False, "answer_reward": -1.,
                "step_scores": [0.]*count, "step_char_ends": [], "absolute_errors": {},
                "step_consistent": [False]*count, "error": str(error)}
    values, reference = parsed["steps"], truth["steps"]
    observed_score, errors = _numeric_score(values[0]["value"], reference[0]["value"], numeric_tolerance_m)
    scores, consistent = [observed_score], [True]
    for i in range(1, count):
        actual, expected = values[i]["value"], reference[i]["value"]
        if isinstance(expected, dict):
            local, more_errors = _numeric_score(actual, expected, numeric_tolerance_m)
            errors.update(more_errors)
        else:
            local = float(actual == expected)
        agrees = _derived_consistent(request, values, i, arithmetic_tolerance_m)
        scores.append(min(scores[-1], local) if agrees else 0.)
        consistent.append(agrees)
    correct = parsed["answer"] == truth["answer"]
    return {"valid": True, "answer": parsed["answer"], "correct": correct,
            "answer_reward": float(correct), "step_scores": scores,
            "step_char_ends": parsed["step_char_ends"], "absolute_errors": errors,
            "step_consistent": consistent, "error": None}
