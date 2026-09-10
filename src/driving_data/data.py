from __future__ import annotations

import concurrent.futures
import hashlib
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .common import BDD_TO_COCO, digest, file_hash, image_path, read_json, write_json

REPO = "dgural/bdd100k"
REVISION = "c2e7f266756bcd07b87f1a45a35937c8eac20241"
INDEX_HASH = "1e8853904e2926c434557bb90a36b1638ac03ebb77adfc21f04d821a6a998a09"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"


def download(url: str, target: Path, expected_hash: str | None = None):
    if target.exists() and (expected_hash is None or file_hash(target) == expected_hash):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        ),
    )
    tmp = target.with_suffix(target.suffix + ".partial")
    try:
        with session.get(url, stream=True, timeout=(20, 90)) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        if expected_hash and file_hash(tmp) != expected_hash:
            raise ValueError("Download checksum mismatch")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
        session.close()


def group_id(sample_id: str) -> str:
    # Conservative filename-prefix grouping, NOT a verified drive/route identity.
    return sample_id.split("-")[0]


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(f"split-v1:{group}".encode()).hexdigest()[:8], 16) % 100
    return "seed" if bucket < 15 else "pool" if bucket < 55 else "dev" if bucket < 75 else "test"


def convert_labels(sample: dict) -> list[dict]:
    w, h = sample["metadata"]["width"], sample["metadata"]["height"]
    result = []
    for obj in sample.get("detections", {}).get("detections", []):
        if obj["label"] not in BDD_TO_COCO:
            continue
        x, y, bw, bh = obj["bounding_box"]
        if not all(math.isfinite(v) for v in (x, y, bw, bh)) or bw <= 0 or bh <= 0:
            raise ValueError("Invalid bounding box")
        box = [max(0.0, x * w), max(0.0, y * h), min(float(w), (x + bw) * w), min(float(h), (y + bh) * h)]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Box outside image")
        result.append(
            {
                "box": box,
                "label": BDD_TO_COCO[obj["label"]],
                "occluded": bool(obj.get("occluded", False)),
                "truncated": bool(obj.get("truncated", False)),
            }
        )
    return result


def validate_partition(rows: list[dict]):
    ids, hashes, groups = set(), {}, {}
    for row in rows:
        if row["sample_id"] in ids:
            raise ValueError("Duplicate sample id")
        ids.add(row["sample_id"])
        if row["sha256"] in hashes:
            raise ValueError("Duplicate image bytes; rebuild manifest after reviewing source")
        hashes[row["sha256"]] = row["split"]
        if row["group_id"] in groups and groups[row["group_id"]] != row["split"]:
            raise ValueError("Group leakage")
        groups[row["group_id"]] = row["split"]


def prepare(root: Path, limit: int = 240, workers: int = 6):
    if limit < 20:
        raise ValueError("Use at least 20 images")
    index = root / "data/raw/samples.json"
    download(f"{BASE}/samples.json", index, INDEX_HASH)
    samples = read_json(index)["samples"]
    if limit > len(samples):
        raise ValueError("Requested more than source size")
    # Select by ID hash only; labels and image quality do not influence membership.
    samples = sorted(samples, key=lambda s: digest(["subset-v1", s["filepath"]]))[:limit]
    rows, errors = [], []

    def process(s):
        sid = Path(s["filepath"]).stem
        if not all(c in "0123456789abcdef-" for c in sid):
            raise ValueError("Unexpected source id")
        target = root / f"data/images/{sid}.jpg"
        download(f"{BASE}/data/{sid}.jpg", target)
        with Image.open(target) as im:
            im.load()
            w, h = im.size
            if (w, h) != (s["metadata"]["width"], s["metadata"]["height"]):
                raise ValueError("Image/annotation dimension mismatch")
            gray = np.asarray(im.convert("L").resize((128, 72)), dtype=float)
        gid = group_id(sid)
        labels = convert_labels(s)
        write_json(root / f"data/labels/{sid}.json", labels)
        return {
            "sample_id": sid,
            "group_id": gid,
            "split": split_for(gid),
            "image_path": f"data/images/{sid}.jpg",
            "sha256": file_hash(target),
            "labels_sha256": digest(labels),
            "width": w,
            "height": h,
            "source_revision": REVISION,
            "weather": s.get("weather", {}).get("label", "unknown"),
            "timeofday": s.get("timeofday", {}).get("label", "unknown"),
            "scene": s.get("scene", {}).get("label", "unknown"),
            "mean_luminance": float(gray.mean()),
            "luminance_std": float(gray.std()),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, s): s["filepath"] for s in samples}
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                errors.append({"source": futures[f], "error_type": type(e).__name__})
            if i % 40 == 0:
                print(f"prepare {i}/{limit}", flush=True)
    write_json(root / "reports/local/ingest_errors.json", errors)
    if errors:
        raise RuntimeError(f"{len(errors)} inputs failed; do not publish a partial dataset. Retry prepare.")
    rows.sort(key=lambda r: r["sample_id"])
    validate_partition(rows)
    version = digest(rows)
    previous = root / "data/manifest.json"
    if previous.exists() and read_json(previous)["version"] != version:
        raise ValueError("Dataset already frozen; use a new project/data root for another subset")
    write_json(previous, {"version": version, "rows": rows})
    pd.DataFrame(rows).to_parquet(root / "data/catalog.parquet", index=False)
    # Sampling receives only these fields. Metadata labels and GT are NOT exported.
    fields = ["sample_id", "group_id", "split", "image_path", "sha256"]
    write_json(root / "data/pool.json", [{k: r[k] for k in fields} for r in rows if r["split"] == "pool"])
    receipt = {
        "dataset_version": version,
        "source": REPO,
        "revision": REVISION,
        "index_sha256": INDEX_HASH,
        "images": len(rows),
        "splits": dict(Counter(r["split"] for r in rows)),
        "source_split": "upstream validation; locally re-split pilot, NOT official benchmark",
        "grouping": "conservative filename prefix; route/driver independence unverified",
        "label_mapping": BDD_TO_COCO,
        "failed_inputs": len(errors),
    }
    write_json(root / "reports/pilot/data_receipt.json", receipt)
    write_json(root / "reports/pilot/sample_manifest.json", {"version": version, "rows": rows})
    return receipt


def verify(root: Path):
    manifest = read_json(root / "data/manifest.json")
    rows = manifest["rows"]
    validate_partition(rows)
    if digest(rows) != manifest["version"]:
        raise ValueError("Manifest checksum mismatch")
    for r in rows:
        if file_hash(image_path(root, r)) != r["sha256"]:
            raise ValueError(f"Image bytes changed: {r['sample_id']}")
        if digest(read_json(root / f"data/labels/{r['sample_id']}.json")) != r["labels_sha256"]:
            raise ValueError(f"Label content changed: {r['sample_id']}")
    return {"verified_images": len(rows), "dataset_version": manifest["version"]}
