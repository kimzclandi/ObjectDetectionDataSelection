from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch

from .common import digest, file_hash, read_json, write_json
from .data import verify
from .evaluation import evaluate
from .model import infer, load_model, tensor_image
from .selection import run_selection


def train(
    root: Path, name: str, rows: list[dict], seed: int, epochs: int, config: dict, initial: Path | None = None
):
    if any(r["split"] not in ("seed", "pool") for r in rows):
        raise ValueError("Evaluation images cannot enter training")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model, model_config = load_model(root, initial, config["device"])
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.roi_heads.box_predictor.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(
        model.roi_heads.box_predictor.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    recipe = {
        "image_ids": sorted(r["sample_id"] for r in rows),
        "label_hashes": {r["sample_id"]: r["labels_sha256"] for r in rows},
        "seed": seed,
        "epochs": epochs,
        "config": config,
        "initial_model": model_config,
    }
    recipe_hash = digest(recipe)
    checkpoint = root / "models" / f"{name}.pth"
    receipt_path = root / "reports/pilot" / name / "training.json"
    if receipt_path.exists() and checkpoint.exists():
        saved = read_json(receipt_path)
        if saved["recipe_sha256"] == recipe_hash and saved["checkpoint_sha256"] == file_hash(checkpoint):
            return checkpoint
        raise ValueError("Existing training artifacts differ; use another run name")
    model.train()
    model.backbone.eval()  # freeze BN running statistics as well as parameters
    rng = random.Random(seed)
    losses, started = [], time.perf_counter()
    for epoch in range(epochs):
        order = list(rows)
        rng.shuffle(order)
        epoch_losses = []
        for row in order:
            gold = read_json(root / f"data/labels/{row['sample_id']}.json")
            if digest(gold) != row["labels_sha256"]:
                raise ValueError("Training label checksum mismatch")
            target = {
                "boxes": torch.tensor(
                    [g["box"] for g in gold], dtype=torch.float32, device=config["device"]
                ).reshape(-1, 4),
                "labels": torch.tensor(
                    [g["label"] for g in gold], dtype=torch.int64, device=config["device"]
                ),
            }
            terms = model([tensor_image(root, row, config["device"])], [target])
            loss = sum(terms.values())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(
            {"epoch": epoch + 1, "mean_total_loss": float(np.mean(epoch_losses)), "steps": len(order)}
        )
        print(f"{name} epoch {epoch + 1}/{epochs} loss={losses[-1]['mean_total_loss']:.4f}", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint.with_suffix(".partial")
    torch.save(model.cpu().state_dict(), tmp)
    tmp.replace(checkpoint)
    receipt = {
        "recipe_sha256": recipe_hash,
        "recipe": recipe,
        "checkpoint_sha256": file_hash(checkpoint),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": time.perf_counter() - started,
        "loss_history": losses,
        "steps": epochs * len(rows),
        "images": len(rows),
        "note": "Actual supervised ROI predictor fine-tuning, not full-detector or VLM training. "
        "Hidden public labels simulate annotation after selection; no human labeling cost measured.",
    }
    write_json(receipt_path, receipt)
    return checkpoint


def regression_gate(baseline, candidate, config):
    old, new = baseline["splits"]["test"], candidate["splits"]["test"]
    night = "timeofday:night"
    a, b = old["slices"].get(night), new["slices"].get(night)
    reasons = []
    if old["coco"]["AP"] is None or new["coco"]["AP"] is None:
        return {"decision": "INSUFFICIENT_EVIDENCE", "reasons": ["No AP support"]}
    delta = new["coco"]["AP"] - old["coco"]["AP"]
    if delta < -config["max_overall_AP_drop"]:
        reasons.append("Overall AP regression")
    supported = (
        a
        and b
        and a["images"] >= config["minimum_night_images"]
        and a["objects"] >= config["minimum_night_objects"]
    )
    if supported and b["recall"] - a["recall"] < -config["max_night_recall_drop"]:
        reasons.append("Night recall regression")
    decision = "REJECT" if reasons else "PASS" if supported else "INSUFFICIENT_EVIDENCE"
    if not supported:
        reasons.append("Night slice below predeclared minimum support")
    return {
        "decision": decision,
        "reasons": reasons,
        "AP_delta": delta,
        "night_recall_delta": b["recall"] - a["recall"] if a and b and a["recall"] is not None else None,
    }


def experiment(root: Path):
    verify(root)
    config = read_json(root / "configs/pilot.json")
    manifest = read_json(root / "data/manifest.json")
    prereg = {
        "config": config,
        "dataset_version": manifest["version"],
        "config_sha256": digest(config),
        "protocol": "Warm-start on seed only; freeze pool inference; 3 paired seeds x 3 samplers. "
        "All candidates start from the SAME warm-start checkpoint. "
        "No test-driven selection or hyperparameter search. Pilot, not significance study.",
    }
    path = root / "reports/pilot/protocol.json"
    if path.exists() and read_json(path) != prereg:
        raise ValueError("Frozen experiment protocol changed")
    write_json(path, prereg)
    seed_rows = [r for r in manifest["rows"] if r["split"] == "seed"]
    initial = train(root, "warmup", seed_rows, config["warmup_seed"], config["warmup_epochs"], config)
    infer(root, "baseline", initial, config["device"])
    baseline = evaluate(root, "baseline")
    run_selection(root, config["budget_images"], tuple(config["strategy_seeds"]))
    plans = read_json(root / "reports/pilot/selections.json")["plans"]
    rows_by_id = {r["sample_id"]: r for r in manifest["rows"]}
    records = []
    for plan in plans:
        name = f"{plan['strategy']}-s{plan['seed']}"
        selected = [rows_by_id[r["sample_id"]] for r in plan["selected"]]
        if any(r["split"] != "pool" for r in selected):
            raise ValueError("Selection contains non-pool samples")
        checkpoint = train(
            root, name, seed_rows + selected, plan["seed"], config["candidate_epochs"], config, initial
        )
        infer(root, name, checkpoint, config["device"], splits=("dev", "test"))
        result = evaluate(root, name)
        gate = regression_gate(baseline, result, config["gate"])
        record = {
            "name": name,
            "strategy": plan["strategy"],
            "seed": plan["seed"],
            "selected_images": len(selected),
            "train_images": len(seed_rows) + len(selected),
            "dev_AP": result["splits"]["dev"]["coco"]["AP"],
            "test_AP": result["splits"]["test"]["coco"]["AP"],
            "gate": gate,
        }
        records.append(record)
        write_json(root / "reports/pilot/experiment_progress.json", records)
    aggregates = []
    for strategy in config["strategies"]:
        values = [r["test_AP"] for r in records if r["strategy"] == strategy]
        paired = [
            r["test_AP"]
            - next(a["test_AP"] for a in records if a["strategy"] == "random" and a["seed"] == r["seed"])
            for r in records
            if r["strategy"] == strategy
        ]
        aggregates.append(
            {
                "strategy": strategy,
                "test_AP_mean": float(np.mean(values)),
                "test_AP_seed_std": float(np.std(values, ddof=1)),
                "paired_AP_delta_vs_random": paired,
                "mean_delta_vs_random": float(np.mean(paired)),
            }
        )
    result = {
        "status": "completed",
        "protocol_sha256": digest(prereg),
        "records": records,
        "aggregates": aggregates,
        "baseline_test_AP": baseline["splits"]["test"]["coco"]["AP"],
        "conclusion_boundary": "Three seeds and a tiny public-data test split are descriptive only. "
        "No measured annotation-cost, safety, full-model or production-scale claim.",
    }
    write_json(root / "reports/pilot/experiment.json", result)
    return result
