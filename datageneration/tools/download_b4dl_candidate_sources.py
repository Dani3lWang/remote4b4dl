#!/usr/bin/env python3
"""Download the public QA/metadata linked by the repository README, without models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import urlopen

REPOSITORY = "ccho4702/nuScenes-B4DL"
# Snapshot inspected during implementation. Use --revision to select another release.
DEFAULT_REVISION = "438bd768356c366ea7e3bd5775e1cbaaa5df7d1b"
FILES = ("README.md", "dataset/train/stage2.json", "dataset/train/stage3.json",
         "dataset/test/existence.json", "dataset/test/binary.json", "dataset/test/time_grounding.json",
         "dataset/test/description.json", "dataset/test/temporal_understanding.json", "dataset/test/comprehensive.json",
         "metadata/scene_metadata.json", "metadata/sequence_metadata.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination: Path, revision: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--revision must be an immutable 40-character commit SHA")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "download_manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else {}
    cached = {row["path"]: row for row in old.get("files", [])} if old.get("revision") == revision else {}
    # Retain previously recorded receipts if interrupted while checking a cache.
    receipts = dict(cached)
    manifest = {"repository": REPOSITORY, "revision": revision, "files": []}
    for relative in FILES:
        target = destination / relative
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{relative}"
        previous = cached.get(relative)
        if previous and target.is_file() and sha256(target) == previous["sha256"]:
            print(f"Verified cache: {relative}")
        else:
            # Preserve unknown or mismatched local files rather than overwriting them.
            if target.exists():
                raise ValueError(f"Unverified existing source file; use a fresh destination: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".download")
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=60) as response, partial.open("wb") as handle:
                        expected = response.headers.get("Content-Length")
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            handle.write(chunk)
                    if expected is not None and partial.stat().st_size != int(expected):
                        raise ValueError(f"Truncated download: {relative}")
                    if relative.endswith(".json"):
                        with partial.open(encoding="utf-8-sig") as handle:
                            if not isinstance(json.load(handle), list):
                                raise ValueError(f"Unexpected source schema: {relative}")
                    partial.replace(target)
                    break
                except (OSError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            print(f"Downloaded: {relative} ({target.stat().st_size} bytes)")
        receipts[relative] = {"path": relative, "url": url, "bytes": target.stat().st_size, "sha256": sha256(target)}
        manifest["files"] = [receipts[name] for name in FILES if name in receipts]
        # Keep successful downloads resumable even if a later request fails.
        part = manifest_path.with_suffix(".json.part")
        part.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        part.replace(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "data/b4dl-official")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    download(args.destination, args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
