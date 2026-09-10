"""Compare a serial scan and Ray Data scans on exactly the same real images.

This intentionally small workload may be slower on Ray; record, don't hide, overhead.
The shared filesystem assumption holds only for the single-host pilot.
"""

import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .common import file_hash, read_json, write_json


def quality_batch(batch):
    ids, hashes, means = [], [], []
    for sid, path in zip(batch["sample_id"], batch["path"]):
        with Image.open(path) as im:
            gray = np.asarray(im.convert("L").resize((128, 72)), dtype=float)
        ids.append(str(sid))
        hashes.append(file_hash(Path(path)))
        means.append(float(gray.mean()))
    return {"sample_id": ids, "sha256": hashes, "mean_luminance": means}


def benchmark(root: Path):
    import ray

    rows = read_json(root / "data/manifest.json")["rows"]
    items = [{"sample_id": r["sample_id"], "path": str((root / r["image_path"]).resolve())} for r in rows]
    started = time.perf_counter()
    serial = quality_batch({k: [r[k] for r in items] for k in ("sample_id", "path")})
    seconds = time.perf_counter() - started
    expected = {
        sid: (sha, mean)
        for sid, sha, mean in zip(serial["sample_id"], serial["sha256"], serial["mean_luminance"])
    }
    if any(expected[r["sample_id"]][0] != r["sha256"] for r in rows):
        raise ValueError("Input bytes changed")
    results = [
        {
            "engine": "serial",
            "workers": 1,
            "images": len(rows),
            "seconds": seconds,
            "images_per_second": len(rows) / seconds,
            "correct": True,
        }
    ]
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    init_start = time.perf_counter()
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        object_store_memory=256 * 1024 * 1024,
        _temp_dir="/private/tmp/driving-ray",
        log_to_driver=False,
    )
    init_seconds = time.perf_counter() - init_start
    try:
        for workers in (1, 2, 4):
            start = time.perf_counter()
            actual = (
                ray.data.from_items(items, override_num_blocks=16)
                .map_batches(quality_batch, batch_format="numpy", batch_size=16, concurrency=workers)
                .take_all()
            )
            elapsed = time.perf_counter() - start
            observed = {r["sample_id"]: (r["sha256"], r["mean_luminance"]) for r in actual}
            if len(actual) != len(expected) or observed != expected:
                raise ValueError("Ray output parity failed")
            results.append(
                {
                    "engine": "ray_data",
                    "workers": workers,
                    "images": len(actual),
                    "seconds": elapsed,
                    "images_per_second": len(actual) / elapsed,
                    "correct": True,
                }
            )
    finally:
        ray.shutdown()
    report = {
        "ray_version": ray.__version__,
        "ray_initialization_seconds": init_seconds,
        "results": results,
        "scope": "Single Apple Silicon host, warm OS file cache, 240 JPEG reads/hash/luminance. "
        "One measured pass per setting, not a stable throughput benchmark. "
        "Scheduling overhead included, Ray init reported separately. No GPU/multi-node/PB claim.",
    }
    write_json(root / "reports/pilot/benchmark.json", report)
    return report
