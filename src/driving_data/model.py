from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.transforms.functional import pil_to_tensor

from .common import CLASSES, digest, file_hash, image_path, read_json, write_json
from .store import Store, runner_lock

WEIGHT_NAME = "fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"


def load_model(root: Path, checkpoint: Path | None = None, device: str = "cpu"):
    torch.set_num_threads(4)
    os.environ["TORCH_HOME"] = str((root / "models").resolve())
    if checkpoint is None:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1, box_score_thresh=0.05
        )
        weight_file = root / "models/hub/checkpoints" / WEIGHT_NAME
    else:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None, box_score_thresh=0.05
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        weight_file = checkpoint
    model.to(device).eval()
    return model, {
        "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "weight_sha256": file_hash(weight_file),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": device,
        "platform": platform.system(),
        "machine": platform.machine(),
        "threads": 4,
        "minimum_score": 0.05,
        "min_size": 320,
        "max_size": 640,
        "classes": list(CLASSES),
        "adapter_version": 1,
    }


def tensor_image(root, row, device):
    with Image.open(image_path(root, row)) as im:
        return pil_to_tensor(im.convert("RGB")).float().div(255).to(device)


def predict_one(model, root, row, device):
    if file_hash(image_path(root, row)) != row["sha256"]:
        raise ValueError("Image checksum changed")
    features = []

    def capture(_module, _inputs, output):
        feature = next(iter(output.values()))
        v = feature.mean(dim=(-2, -1))[0].detach().cpu().numpy()
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        features.extend(v.astype(float).tolist())

    handle = model.backbone.register_forward_hook(capture)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            out = model([tensor_image(root, row, device)])[0]
        boxes, scores, labels = (out[k].detach().cpu().tolist() for k in ("boxes", "scores", "labels"))
    finally:
        handle.remove()
    predictions = [
        {"box": b, "score": float(s), "label": int(c)}
        for b, s, c in zip(boxes, scores, labels)
        if c in CLASSES
    ]
    return {
        "sample_id": row["sample_id"],
        "sha256": row["sha256"],
        "predictions": predictions,
        "embedding": features,
        "latency_seconds": time.perf_counter() - started,
    }


def infer(
    root: Path,
    name="baseline",
    checkpoint: Path | None = None,
    device="cpu",
    splits: tuple[str, ...] = ("seed", "pool", "dev", "test"),
    crash_after: int | None = None,
):
    manifest = read_json(root / "data/manifest.json")
    rows = [r for r in manifest["rows"] if r["split"] in splits]
    if not rows:
        raise ValueError("No images in requested splits")
    model, config = load_model(root, checkpoint, device)
    run_id = digest([manifest["version"], config, sorted(splits)])
    outdir = root / "reports/pilot" / name
    dbpath = root / "work/jobs" / f"{run_id}.sqlite"
    output, hits, computed = [], 0, 0
    started = time.perf_counter()
    with runner_lock(dbpath):
        store = Store(dbpath)
        try:
            recovered = store.recover()
            for row in rows:
                key = digest([row["sample_id"], row["sha256"], config])
                store.enqueue(key, row["sample_id"])
                # Verify even cache hits; corrupted inputs must not silently pass.
                if file_hash(image_path(root, row)) != row["sha256"]:
                    raise ValueError("Image checksum mismatch")
                cached = store.get(key)
                if cached:
                    output.append(cached)
                    hits += 1
                    continue
                for attempt in range(2):
                    store.start(key)
                    if crash_after is not None and computed == crash_after:
                        raise SystemExit("Injected driver interruption after claim, before result commit")
                    try:
                        result = predict_one(model, root, row, device)
                        store.succeed(key, result)
                        output.append(result)
                        computed += 1
                        break
                    except Exception as e:
                        store.fail(key, type(e).__name__)
                        if attempt == 1:
                            raise
                if len(output) % 40 == 0:
                    print(f"{name}: {len(output)}/{len(rows)}", flush=True)
            counts = store.counts()
        finally:
            store.close()
    seconds = time.perf_counter() - started
    payload = {"run_id": run_id, "dataset_version": manifest["version"], "config": config, "rows": output}
    write_json(outdir / "predictions.json", payload)
    receipt = {
        "run_id": run_id,
        "images": len(rows),
        "computed": computed,
        "cache_hits": hits,
        "recovered_jobs": recovered,
        "job_counts": counts,
        "seconds_excluding_model_load": seconds,
        "effective_images_per_second": len(rows) / seconds,
        "p50_compute_seconds": float(np.median([r["latency_seconds"] for r in output])),
        "p95_compute_seconds": float(np.percentile([r["latency_seconds"] for r in output], 95)),
        "latency_note": "Per-image original compute latency; throughput includes cache hits when present.",
    }
    write_json(outdir / "run_receipt.json", receipt)
    return receipt
