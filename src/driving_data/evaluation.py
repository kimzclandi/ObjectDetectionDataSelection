from __future__ import annotations

import contextlib
import io
import math
from collections import Counter
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .common import CLASSES, digest, read_json, write_json


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def validate_objects(objects, *, predictions=False):
    """Reject malformed numeric records; an empty prediction list is a valid miss."""
    if not isinstance(objects, list):
        raise ValueError("Objects must be a list")
    for obj in objects:
        if not isinstance(obj, dict) or type(obj.get("label")) is not int or obj["label"] not in CLASSES:
            raise ValueError("Unsupported object label")
        box = obj.get("box")
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
        ):
            raise ValueError("Box must contain four finite coordinates")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Box must have positive width and height")
        if predictions:
            score = obj.get("score")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Prediction score must be finite and within [0, 1]")


def match_objects(gold, predictions, threshold=0.5):
    """Score-ordered, class-aware one-to-one matching at IoU >= 0.5."""
    validate_objects(gold)
    validate_objects(predictions, predictions=True)
    used, matches, fp = set(), [], 0
    for p in sorted(predictions, key=lambda x: -x["score"]):
        if p["score"] < threshold:
            continue
        candidates = [
            (iou(g["box"], p["box"]), j)
            for j, g in enumerate(gold)
            if j not in used and g["label"] == p["label"]
        ]
        best = max(candidates, default=(0.0, -1))
        if best[0] >= 0.5:
            used.add(best[1])
            matches.append(best[1])
        else:
            fp += 1
    return {"tp": len(matches), "fp": fp, "fn": len(gold) - len(matches), "matched_indices": sorted(matches)}


def coco_metrics(rows, labels, predictions):
    ids = [r["sample_id"] for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Evaluation image IDs must be nonempty and unique")
    if any(sid not in labels or sid not in predictions for sid in ids):
        raise ValueError("Missing evaluation labels or predictions")
    for sid in ids:
        validate_objects(labels[sid])
        validate_objects(predictions[sid]["predictions"], predictions=True)
    annotations, detections = [], []
    for image_id, r in enumerate(rows, 1):
        for g in labels[r["sample_id"]]:
            x1, y1, x2, y2 = g["box"]
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": g["label"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
        for p in predictions[r["sample_id"]]["predictions"]:
            x1, y1, x2, y2 = p["box"]
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": p["label"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": p["score"],
                }
            )
    with contextlib.redirect_stdout(io.StringIO()):
        gold = COCO()
        gold.dataset = {
            "info": {},
            "images": [{"id": i, "width": r["width"], "height": r["height"]} for i, r in enumerate(rows, 1)],
            "categories": [{"id": k, "name": v} for k, v in CLASSES.items()],
            "annotations": annotations,
        }
        gold.createIndex()
        if detections:
            pred = gold.loadRes(detections)
        else:
            pred = COCO()
            pred.dataset = {**gold.dataset, "annotations": []}
            pred.createIndex()
        evaluator = COCOeval(gold, pred, "bbox")
        evaluator.params.catIds = list(CLASSES)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    keys = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return {k: float(v) if v >= 0 else None for k, v in zip(keys, evaluator.stats)}


def evaluate(root: Path, name="baseline", splits=("dev", "test"), output: Path | None = None):
    manifest = read_json(root / "data/manifest.json")
    prediction_file = root / "reports/pilot" / name / "predictions.json"
    payload = read_json(prediction_file)
    if payload["dataset_version"] != manifest["version"]:
        raise ValueError("Prediction/dataset version mismatch")
    preds = {r["sample_id"]: r for r in payload["rows"]}
    if len(preds) != len(payload["rows"]):
        raise ValueError("Duplicate predictions")
    reports, all_samples = {}, []
    for split in splits:
        rows = [r for r in manifest["rows"] if r["split"] == split]
        if not rows:
            raise ValueError("Empty evaluation split")
        if any(r["sample_id"] not in preds or preds[r["sample_id"]]["sha256"] != r["sha256"] for r in rows):
            raise ValueError("Missing or stale predictions: failed images may not be dropped")
        labels = {r["sample_id"]: read_json(root / f"data/labels/{r['sample_id']}.json") for r in rows}
        if any(digest(labels[r["sample_id"]]) != r["labels_sha256"] for r in rows):
            raise ValueError("Evaluation labels changed after dataset freeze")
        slices = {}
        for r in rows:
            sid = r["sample_id"]
            gold = labels[sid]
            m = match_objects(gold, preds[sid]["predictions"])
            matched = set(m.pop("matched_indices"))
            sample = {
                "sample_id": sid,
                "split": split,
                "timeofday": r["timeofday"],
                "weather": r["weather"],
                "scene": r["scene"],
                "objects": len(gold),
                **m,
            }
            sample["recall"] = m["tp"] / len(gold) if gold else None
            all_samples.append(sample)
            tags = ["all", f"timeofday:{r['timeofday']}", f"weather:{r['weather']}"]
            for tag in tags:
                c = slices.setdefault(tag, Counter())
                c.update({"images": 1, "objects": len(gold), **m})
            for tag, fn in [
                ("object:small", lambda g: (g["box"][2] - g["box"][0]) * (g["box"][3] - g["box"][1]) < 32**2),
                ("object:occluded", lambda g: g["occluded"]),
                ("object:person", lambda g: g["label"] == 1),
            ]:
                indices = {i for i, g in enumerate(gold) if fn(g)}
                c = slices.setdefault(tag, Counter())
                c.update(
                    {
                        "images": int(bool(indices)),
                        "objects": len(indices),
                        "tp": len(indices & matched),
                        "fn": len(indices - matched),
                    }
                )
        for c in slices.values():
            c["recall"] = c["tp"] / c["objects"] if c["objects"] else None
            c["precision"] = c["tp"] / (c["tp"] + c["fp"]) if "fp" in c and c["tp"] + c["fp"] else None
        reports[split] = {"images": len(rows), "coco": coco_metrics(rows, labels, preds), "slices": slices}
    result = {
        "run_id": payload["run_id"],
        "dataset_version": manifest["version"],
        "splits": reports,
        "operating_point": {"score_threshold": 0.5, "iou_threshold": 0.5},
        "metric_note": "COCO bbox AP on seven mapped classes, not official BDD metric. "
        "Small = original-image area < 32^2 pixels. Object slices report recall only. "
        "No safety or population-generalization claim.",
    }
    destination = output if output is not None else prediction_file.parent
    artifacts = {"evaluation.json": result, "sample_metrics.json": all_samples}
    # Validate both before writing either. Identical replay is read-only; a changed
    # result requires an explicitly separate directory, never a frozen-file rewrite.
    for filename, value in artifacts.items():
        path = destination / filename
        if path.exists() and read_json(path) != value:
            raise ValueError(f"Existing evaluation differs: {path}. Use --output with a new directory.")
    for filename, value in artifacts.items():
        path = destination / filename
        if not path.exists():
            write_json(path, value)
    return result
