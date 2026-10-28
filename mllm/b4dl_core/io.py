"""Small deterministic artifact helpers; no model or dataset dependency."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


@contextmanager
def atomic_output(path, mode="w"):
    """Publish only a flushed, complete file, on the same filesystem."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        options = {} if "b" in mode else {"encoding": "utf-8", "newline": "\n"}
        with os.fdopen(fd, mode, **options) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    with atomic_output(path) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                allow_nan=False, indent=2) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, values):
    with atomic_output(path) as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def unique_index(rows, key="token"):
    result = {}
    for row in rows:
        ident = row[key]
        if not isinstance(ident, str) or not ident or ident in result:
            raise ValueError(f"empty or duplicate {key}: {ident!r}")
        result[ident] = row
    return result


def safe_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"path outside root: {name}")
    return path


def iter_array(path):
    """Stream nuScenes' large JSON arrays using bounded buffers (stdlib only)."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8-sig") as stream:
        buffer, eof = "", False

        def fill():
            nonlocal buffer, eof
            chunk = stream.read(65536)
            buffer += chunk
            eof = not chunk

        def space():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()

        space()
        if not buffer.startswith("["):
            raise ValueError("expected an array: " + str(path))
        buffer = buffer[1:]
        space()
        first = True
        while not buffer.startswith("]"):
            if not first:
                if not buffer.startswith(","):
                    raise ValueError("missing array separator")
                buffer = buffer[1:]
                space()
                if buffer.startswith("]"):
                    raise ValueError("trailing comma")
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError("invalid/truncated JSON: " + str(path)) from None
                    fill()
            if not isinstance(value, dict):
                raise ValueError("expected object rows")
            yield value
            buffer = buffer[end:]
            first = False
            space()
        buffer = buffer[1:]
        space()
        if buffer:
            raise ValueError("trailing JSON content")
