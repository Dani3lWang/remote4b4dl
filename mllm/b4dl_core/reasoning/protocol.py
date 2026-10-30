"""Closed, task-specific reasoning traces; no generated code is executed."""
import json
import math

from ..geometry import SECTORS
from ..rewards import _bad_constant, _pairs


class InvalidTrace(ValueError):
    """A model completion does not satisfy the public trace contract."""


def template(request):
    task = request["task"]
    if task == "existence":
        return [("observe", ("count",)), ("exists", None)]
    if task == "position":
        return [("observe", ("x", "y")), ("sector", None), ("predicate", None)]
    if task == "distance":
        return [("observe", tuple(sorted("distance_" + v for v in request["options"].values()))),
                ("argmin", None)]
    if task == "motion":
        return [("observe", ("d0", "d1", "y0", "y1")),
                ("delta", ("delta_d", "delta_y")), ("predicate", None)]
    raise ValueError("unsupported reasoning task")


def instruction(request):
    specs = template(request)
    first = ', '.join(specs[0][1])
    operations = ', '.join(op for op, _ in specs)
    meaning = {
        "existence": "count is the number of LiDAR-supported objects of the requested class in the specified frame.",
        "position": "x and y are the nearest requested object's box-centre coordinates in the frame's ego system; x points forward and y points left.",
        "distance": "Each distance_CLASS is the ground-plane distance from ego to the nearest LiDAR-supported object of that option's class.",
        "motion": "d0 and d1 are ground-plane distances, and y0 and y1 are lateral coordinates, at the first and second frames respectively; each uses that frame's ego system with y pointing left. The delta value has numeric keys delta_d=d1-d0 and delta_y=y1-y0.",
    }[request["task"]]
    answer = ("The answer must be the JSON string \"Yes\" or \"No\"." if request["kind"] == "binary"
              else "The answer must be the JSON string containing one listed option letter.")
    return ("Return only a JSON object with keys in this order: steps, answer. "
            "The steps array must contain exactly these operations in order: " + operations + ". "
            "The first step is {\"op\":\"observe\",\"frames\":"
            + json.dumps(request["frames"], separators=(",", ":"))
            + ",\"value\":{...}} with exactly these numeric value keys: " + first + ". "
            "Each subsequent step is {\"op\":...,\"depends_on\":[previous step index],\"value\":...}; "
            "step indexes start at 0. Use metres, integer count, and JSON booleans for exists/predicate. "
            "Use one of front, front-left, left, back-left, back, back-right, right, front-right for sector; "
            "use an option letter for argmin. " + meaning + " " + answer)


def public_prompt(request):
    text = request["question"]
    if request["options"]:
        text += "\n" + "\n".join(f"{k}. {v}" for k, v in request["options"].items())
    return text + "\n" + instruction(request)


def encode_trace(trace):
    return json.dumps(trace, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _space(text, i):
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def _literal(text, i, expected):
    i = _space(text, i)
    if not text.startswith(expected, i):
        raise InvalidTrace("expected " + expected)
    return i + len(expected)


def _numeric_map(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        return False
    for key, number in value.items():
        if type(number) not in (int, float) or not math.isfinite(number):
            return False
        if key == "count" and (type(number) is not int or number < 0):
            return False
        if (key.startswith("distance_") or key in ("d0", "d1")) and number < 0:
            return False
    return True


def parse_trace(text, request):
    """Parse exact steps-first JSON and return character ends for complete steps.

    Duplicate keys, extra fields and extra/truncated content are rejected. The
    parser records offsets while decoding the original text, not by re-encoding.
    """
    if not isinstance(text, str):
        raise InvalidTrace("completion must be text")
    decoder = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_bad_constant)
    try:
        i = _literal(text, 0, "{")
        i = _literal(text, i, '"steps"')
        i = _literal(text, i, ":")
        i = _literal(text, i, "[")
        steps, ends = [], []
        specs = template(request)
        for n, (op, keys) in enumerate(specs):
            if n:
                i = _literal(text, i, ",")
            i = _space(text, i)
            step, i = decoder.raw_decode(text, i)
            if not isinstance(step, dict) or step.get("op") != op:
                raise InvalidTrace("unexpected step operation")
            expected = {"op", "frames", "value"} if n == 0 else {"op", "depends_on", "value"}
            if set(step) != expected:
                raise InvalidTrace("unexpected step fields")
            if n == 0:
                if step["frames"] != request["frames"] or any(type(v) is not int for v in step["frames"]):
                    raise InvalidTrace("incorrect frame reference")
            elif step["depends_on"] != [n-1] or any(type(v) is not int for v in step["depends_on"]):
                raise InvalidTrace("incorrect step dependency")
            value = step["value"]
            if keys is not None and not _numeric_map(value, keys):
                raise InvalidTrace("invalid numeric value")
            if op in ("exists", "predicate") and type(value) is not bool:
                raise InvalidTrace("predicate must be boolean")
            if op == "sector" and (not isinstance(value, str) or value not in SECTORS):
                raise InvalidTrace("invalid sector")
            if op == "argmin" and (not isinstance(value, str) or value not in request["options"]):
                raise InvalidTrace("invalid option")
            steps.append(step)
            ends.append(i)
        i = _literal(text, i, "]")
        i = _literal(text, i, ",")
        i = _literal(text, i, '"answer"')
        i = _literal(text, i, ":")
        answer, i = decoder.raw_decode(text, _space(text, i))
        i = _literal(text, i, "}")
        if _space(text, i) != len(text):
            raise InvalidTrace("trailing content")
        allowed = set(request["options"]) if request["kind"] == "mcq" else {"Yes", "No"}
        if not isinstance(answer, str) or answer not in allowed:
            raise InvalidTrace("invalid answer")
        return {"steps": steps, "answer": answer, "step_char_ends": ends}
    except InvalidTrace:
        raise
    except (ValueError, TypeError, OverflowError, RecursionError, KeyError, IndexError) as error:
        raise InvalidTrace("malformed trace") from error


def make_trace(request, target, endpoints=None):
    """Construct a deterministic teacher trace from private verified truth."""
    from ..geometry import motion_truth, sector
    evidence, task = target["evidence"], request["task"]
    if task == "existence":
        observed, derived = {"count": evidence["count"]}, [("exists", evidence["count"] > 0)]
    elif task == "position":
        observed = {k: evidence[k] for k in ("x", "y")}
        location = sector(observed["x"], observed["y"])
        derived = [("sector", location), ("predicate", location == request["predicate"])]
    elif task == "distance":
        observed = {k: evidence[k] for k in sorted(evidence)}
        derived = [("argmin", target["answer"])]
    elif task == "motion":
        if endpoints is None or set(endpoints) != {"y0", "y1"}:
            raise ValueError("verified endpoint y coordinates required for motion")
        observed = {"d0": evidence["d0"], "d1": evidence["d1"], **endpoints}
        delta = {"delta_d": evidence["delta_d"], "delta_y": evidence["delta_y"]}
        derived = [("delta", delta), ("predicate", motion_truth(delta)[request["predicate"]])]
    else:
        raise ValueError("unsupported reasoning task")
    steps = [{"op": "observe", "frames": request["frames"], "value": observed}]
    for op, value in derived:
        steps.append({"op": op, "depends_on": [len(steps)-1], "value": value})
    trace = {"steps": steps, "answer": target["answer"]}
    parse_trace(encode_trace(trace), request)
    return trace
