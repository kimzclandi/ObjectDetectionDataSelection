from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

CLASSES = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck", 10: "traffic light"}
BDD_TO_COCO = {
    "pedestrian": 1,
    "person": 1,
    "rider": 1,
    "bicycle": 2,
    "bike": 2,
    "car": 3,
    "motorcycle": 4,
    "motor": 4,
    "bus": 6,
    "truck": 8,
    "traffic light": 10,
}


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".write-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path):
    return json.loads(path.read_text())


def image_path(root: Path, row: dict) -> Path:
    path = (root / row["image_path"]).resolve()
    if not path.is_relative_to((root / "data/images").resolve()):
        raise ValueError("Image path escapes data/images")
    return path
