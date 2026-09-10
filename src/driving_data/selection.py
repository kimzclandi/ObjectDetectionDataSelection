from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from .common import digest, read_json, write_json

POOL_FIELDS = {"sample_id", "group_id", "split", "image_path", "sha256"}


def uncertainty(predictions):
    # Post-NMS score entropy is a heuristic, not calibrated epistemic uncertainty.
    scores = sorted([p["score"] for p in predictions], reverse=True)[:20]
    if not scores:
        return 1.0  # explicit exploration policy for no detections, NOT proof of uncertainty
    return float(
        np.mean(
            [
                -(p * math.log(max(p, 1e-12)) + (1 - p) * math.log(max(1 - p, 1e-12))) / math.log(2)
                for p in scores
            ]
        )
    )


def select(pool, predictions, budget, strategy, seed):
    if not isinstance(budget, int) or budget <= 0 or budget > len(pool):
        raise ValueError("Budget outside pool")
    if any(set(r) != POOL_FIELDS or r["split"] != "pool" for r in pool):
        raise ValueError("Sampler accepts pool-only unlabeled contract; metadata/GT forbidden")
    if len({r["sample_id"] for r in pool}) != len(pool):
        raise ValueError("Duplicate pool IDs")
    lookup = {r["sample_id"]: r for r in predictions}
    if any(r["sample_id"] not in lookup or lookup[r["sample_id"]]["sha256"] != r["sha256"] for r in pool):
        raise ValueError("Missing or stale pool predictions")
    rng = random.Random(seed)
    candidates = sorted(pool, key=lambda r: r["sample_id"])
    rng.shuffle(candidates)
    factors = {r["sample_id"]: uncertainty(lookup[r["sample_id"]]["predictions"]) for r in candidates}
    if strategy == "random":
        chosen = candidates[:budget]
    elif strategy == "uncertainty":
        chosen = sorted(candidates, key=lambda r: -factors[r["sample_id"]])[:budget]
    elif strategy == "diverse":
        # Greedy normalized entropy + minimum cosine distance; at most one per group.
        if len({r["group_id"] for r in candidates}) < budget:
            raise ValueError("Not enough unique groups for diversity budget")
        chosen, groups = [], set()
        vectors = {}
        for r in candidates:
            v = np.asarray(lookup[r["sample_id"]]["embedding"], dtype=float)
            if v.ndim != 1 or not v.size or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-12:
                raise ValueError("Invalid image embedding")
            vectors[r["sample_id"]] = v / np.linalg.norm(v)
        while len(chosen) < budget:

            def score(r):
                distance = (
                    min((1 - float(vectors[r["sample_id"]] @ vectors[c["sample_id"]])) / 2 for c in chosen)
                    if chosen
                    else 0.0
                )
                return 0.5 * factors[r["sample_id"]] + 0.5 * distance

            best = max((r for r in candidates if r["group_id"] not in groups), key=score)
            chosen.append(best)
            groups.add(best["group_id"])
    else:
        raise ValueError("Unknown strategy")
    return [
        {
            "sample_id": r["sample_id"],
            "rank": i,
            "uncertainty_proxy": factors[r["sample_id"]],
            "no_detections": not lookup[r["sample_id"]]["predictions"],
            "group_id": r["group_id"],
        }
        for i, r in enumerate(chosen, 1)
    ]


def run_selection(root: Path, budget=12, seeds=(17, 29, 43)):
    pool = read_json(root / "data/pool.json")
    baseline = read_json(root / "reports/pilot/baseline/predictions.json")
    manifest = read_json(root / "data/manifest.json")
    if baseline["dataset_version"] != manifest["version"]:
        raise ValueError("Dataset mismatch")
    plans = []
    for seed in seeds:
        for strategy in ("random", "uncertainty", "diverse"):
            selected = select(pool, baseline["rows"], budget, strategy, seed)
            plans.append({"strategy": strategy, "seed": seed, "budget_images": budget, "selected": selected})
    result = {
        "dataset_version": manifest["version"],
        "inference_run_id": baseline["run_id"],
        "pool_contract_sha256": digest(pool),
        "plans": plans,
        "note": "No GT or source scene labels used. Equal IMAGE budget, not measured human cost. "
        "Deterministic uncertainty/diverse sets can be identical across seeds; training seeds vary.",
    }
    write_json(root / "reports/pilot/selections.json", result)
    pred_by_id = {r["sample_id"]: r for r in baseline["rows"]}
    selected_by = {}
    for plan in plans:
        for sample in plan["selected"]:
            selected_by.setdefault(sample["sample_id"], []).append(f"{plan['strategy']}-s{plan['seed']}")
    queue = [
        {
            "sample_id": sid,
            "status": "unreviewed",
            "model_run_id": baseline["run_id"],
            "selected_by": selected_by[sid],
            "proposed_boxes": [p for p in pred_by_id[sid]["predictions"] if p["score"] >= 0.5],
            "note": "Model proposals only; not ground truth and not used as training labels in this pilot.",
        }
        for sid in sorted(selected_by)
    ]
    write_json(root / "reports/pilot/review_queue.json", queue)
    return {"plans": len(plans), "budget_images": budget}
