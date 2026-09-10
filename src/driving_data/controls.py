"""Post-hoc, matched-step no-new-data ablation, kept separate from the original pilot."""

import random
from pathlib import Path

import numpy as np

from .common import digest, read_json, write_json
from .data import verify
from .evaluation import evaluate
from .model import infer
from .training import train


def matched_step_rows(seed_rows, extra_slots, seed):
    if not seed_rows or any(r["split"] != "seed" for r in seed_rows):
        raise ValueError("Control accepts only seed data")
    if extra_slots < 0 or extra_slots > len(seed_rows):
        raise ValueError("Unsupported repeat budget")
    return seed_rows + random.Random(seed).sample(seed_rows, extra_slots)


def controls(root: Path):
    verify(root)
    config = read_json(root / "configs/pilot.json")
    manifest = read_json(root / "data/manifest.json")
    experiment = read_json(root / "reports/pilot/experiment.json")
    protocol = {
        "kind": "post_hoc_followup",
        "dataset_version": manifest["version"],
        "pilot_config_sha256": digest(config),
        "design": "44 unique seed images + 12 deterministic repeated seed slots per epoch; "
        "2 epochs = 112 steps. Same warmup checkpoint, optimizer and three seeds. "
        "No pool image or additional label consumed. Designed after viewing pilot results; "
        "descriptive ablation on reused public test, not fresh confirmatory evidence.",
    }
    path = root / "reports/pilot/control_protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("Control protocol changed")
    write_json(path, protocol)
    seed_rows = [r for r in manifest["rows"] if r["split"] == "seed"]
    records = []
    for seed in config["strategy_seeds"]:
        rows = matched_step_rows(seed_rows, config["budget_images"], seed)
        name = f"seed-only-s{seed}"
        checkpoint = train(
            root, name, rows, seed, config["candidate_epochs"], config, root / "models/warmup.pth"
        )
        infer(root, name, checkpoint, config["device"], splits=("dev", "test"))
        result = evaluate(root, name)
        ap = result["splits"]["test"]["coco"]["AP"]
        records.append(
            {
                "name": name,
                "seed": seed,
                "unique_images": len(seed_rows),
                "new_images": 0,
                "image_slots_per_epoch": len(rows),
                "steps": len(rows) * config["candidate_epochs"],
                "test_AP": ap,
                "dev_AP": result["splits"]["dev"]["coco"]["AP"],
                "random_minus_control_AP": next(
                    r["test_AP"]
                    for r in experiment["records"]
                    if r["strategy"] == "random" and r["seed"] == seed
                )
                - ap,
            }
        )
    report = {
        "protocol": protocol,
        "records": records,
        "test_AP_mean": float(np.mean([r["test_AP"] for r in records])),
        "test_AP_seed_std": float(np.std([r["test_AP"] for r in records], ddof=1)),
        "random_minus_control_AP_mean": float(np.mean([r["random_minus_control_AP"] for r in records])),
    }
    write_json(root / "reports/pilot/controls.json", report)
    return report
