"""Durable inference records and conservative recovery of interrupted appends."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re

from .io import atomic_output, read_json, sha256, write_json


@contextmanager
def output_lease(output):
    """OS-owned lock: a killed writer releases it without deleting any files."""
    with (Path(output) / ".inference.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def bind_run(path, run, resume, artifacts):
    """Never adopt unidentified predictions, including historical partial runs."""
    path = Path(path)
    if path.exists():
        if not resume:
            raise ValueError("existing inference requires explicit resume")
        if read_json(path) != run:
            raise ValueError("resume prediction run identity mismatch")
    else:
        if resume or any(Path(p).exists() for p in artifacts):
            raise ValueError("missing resume identity; refuse to adopt existing predictions")
        write_json(path, run)


def validate_predictions(rows, predictions, complete=False):
    if len(predictions) > len(rows) or (complete and len(predictions) != len(rows)):
        raise ValueError("incomplete or extra prediction rows")
    for row, prediction in zip(rows, predictions):
        if not isinstance(prediction, dict) or any(prediction.get(k) != row[k]
                for k in ("request_id", "input_fingerprint")):
            raise ValueError("prediction order/identity mismatch")
        if prediction.get("status") not in ("ok", "truncated", "error", "unavailable") or not isinstance(prediction.get("text"), str):
            raise ValueError("invalid prediction payload")


class _Incomplete(Exception):
    pass


def incomplete_object(text):
    """True only for a syntactically valid, unfinished JSON object prefix.

    JSONDecodeError alone cannot distinguish truncation from corrupt content.
    This small grammar rejects malformed interior tokens and trailing garbage.
    """
    position = 0

    def peek():
        nonlocal position
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
        if position == len(text):
            raise _Incomplete
        return text[position]

    def string():
        nonlocal position
        if peek() != '"':
            raise ValueError
        position += 1
        while True:
            if position == len(text):
                raise _Incomplete
            char = text[position]
            position += 1
            if char == '"':
                return
            if ord(char) < 32:
                raise ValueError
            if char == "\\":
                if position == len(text):
                    raise _Incomplete
                char = text[position]
                position += 1
                if char == "u":
                    for _ in range(4):
                        if position == len(text):
                            raise _Incomplete
                        if text[position] not in "0123456789abcdefABCDEF":
                            raise ValueError
                        position += 1
                elif char not in '"\\/bfnrt':
                    raise ValueError

    def value():
        nonlocal position
        char = peek()
        if char == '"':
            string()
        elif char in "{[":
            opening = char
            closing = "}" if char == "{" else "]"
            position += 1
            if peek() == closing:
                position += 1
                return
            while True:
                if opening == "{":
                    string()
                    if peek() != ":":
                        raise ValueError
                    position += 1
                value()
                char = peek()
                position += 1
                if char == closing:
                    return
                if char != ",":
                    raise ValueError
        elif char in "tfn":
            literal = {"t": "true", "f": "false", "n": "null"}[char]
            for expected in literal:
                if position == len(text):
                    raise _Incomplete
                if text[position] != expected:
                    raise ValueError
                position += 1
        else:
            tail = text[position:]
            match = re.match(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", tail)
            if re.fullmatch(r"-|(?:-?(?:0|[1-9][0-9]*))(?:\.|(?:\.[0-9]+)?[eE][+-]?)", tail):
                raise _Incomplete
            if not match:
                raise ValueError
            position += match.end()

    if not text.lstrip().startswith("{"):
        return False
    try:
        value()
    except _Incomplete:
        return True
    except (ValueError, RecursionError):
        return False
    return False


def recover_jsonl(path, validate):
    """Validate before repair; archive exact original bytes before changing them."""
    path = Path(path)
    content = path.read_bytes() if path.exists() else b""
    lines = content.split(b"\n")
    rows, prefix, repair = [], 0, False
    for index, raw in enumerate(lines):
        last = index == len(lines) - 1
        if last and not raw:
            break
        try:
            line = raw.decode("utf-8-sig" if index == 0 else "utf-8")
        except UnicodeDecodeError as error:
            if not last or error.reason != "unexpected end of data":
                raise ValueError("corrupt JSONL encoding") from error
            # A torn UTF-8 codepoint is legal only inside an unfinished string.
            line = raw[:error.start].decode("utf-8-sig" if index == 0 else "utf-8") + "x"
            if not incomplete_object(line):
                raise ValueError("corrupt JSONL tail") from error
            repair = True
            break
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if not last or not incomplete_object(line):
                raise ValueError("corrupt JSONL record; refusing recovery") from error
            repair = True
            break
        rows.append(row)
        prefix += len(raw) + (0 if last else 1)
        repair = last
    validate(rows)
    if repair:
        backup = path.with_name(path.name + ".before-recovery-" + hashlib.sha256(content).hexdigest() + ".bak")
        if backup.exists():
            if backup.read_bytes() != content:
                raise ValueError("recovery backup mismatch")
        else:
            with atomic_output(backup, "wb") as stream:
                stream.write(content)
        repaired = content[:prefix]
        if repaired and not repaired.endswith(b"\n"):
            repaired += b"\n"
        with atomic_output(path, "wb") as stream:
            stream.write(repaired)
    return rows


def append_jsonl(path, values):
    # Serialize the entire transaction before touching the existing file.
    content = "".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in values).encode("utf-8")
    with Path(path).open("ab") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def verify_completion(output, expected, files):
    path = Path(output) / "completion.json"
    if not path.exists():
        return False
    actual = read_json(path)
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("completion identity/count mismatch")
    for key, file in files.items():
        if not Path(file).is_file() or actual.get(key) != sha256(file):
            raise ValueError("completed inference file hash mismatch")
    return True
