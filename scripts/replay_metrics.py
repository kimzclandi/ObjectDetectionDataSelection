"""Recompute saved failure-v2 metrics from labels, without images, weights or inference."""

import argparse
from pathlib import Path

from driving_data.common import digest, file_hash, read_json, write_json
from driving_data.failure_loop import OUT, SEEDS, metrics

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/maintenance/2026-09-19/baseline"


def replay(labels_dir):
    protocol = read_json(OUT / "protocol.json")
    manifest = read_json(OUT / "manifest.json")
    if digest(manifest) != protocol["manifest"]:
        raise ValueError("Frozen evaluation manifest changed")
    for name, expected in protocol["code"].items():
        path = (
            ARCHIVE / f"{name}.txt"
            if name
            in {
                "src/driving_data/cli.py",
                "src/driving_data/evaluation.py",
            }
            else ROOT / name
        )
        if file_hash(path) != expected:
            raise ValueError(f"Historical source identity mismatch: {name}")
    for name, expected in read_json(OUT / "artifact_manifest.json").items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Frozen evidence changed: {name}")
    rows = [row for row in manifest["rows"] if row["split"] == "test"]
    if len(rows) != 120 or len({row["sample_id"] for row in rows}) != 120:
        raise ValueError("Expected 120 unique evaluation images")
    labels = {}
    for row in rows:
        path = labels_dir / f"{row['sample_id']}.json"
        if not path.is_file():
            raise ValueError(f"Missing reference labels: {path}; see docs/maintenance/2026-09-19/README.md")
        value = read_json(path)
        if digest(value) != row["labels_sha256"]:
            raise ValueError(f"Reference label identity mismatch: {row['sample_id']}")
        labels[row["sample_id"]] = value
    by_id = {row["sample_id"]: row for row in rows}
    all_predictions, results = {}, {}
    for name in ["baseline"] + [
        f"{arm}-s{seed}" for seed in SEEDS for arm in ("targeted", "random", "seed_only")
    ]:
        predictions = {}
        for path in sorted((OUT / "runs" / name / "predictions").glob("*.json")):
            record = read_json(path)
            prediction = record["prediction"]
            sid = prediction["sample_id"]
            if sid not in by_id or sid in predictions or prediction["sha256"] != by_id[sid]["sha256"]:
                raise ValueError("Duplicate, unexpected or stale prediction")
            expected = digest([digest(protocol), record["model"], sid, by_id[sid]["sha256"]])
            if record["key"] != expected:
                raise ValueError("Prediction identity mismatch")
            predictions[sid] = prediction
        if set(predictions) != set(by_id):
            raise ValueError("Missing predictions may not disappear from the denominator")
        all_predictions[name] = predictions
        computed = metrics(rows, labels, predictions, all_predictions["baseline"])
        if computed != read_json(OUT / "runs" / name / "evaluation.json"):
            raise ValueError(f"Recomputed metrics differ: {name}")
        results[name] = computed
    means = {
        arm: sum(results[f"{arm}-s{seed}"]["coco"]["AP"] for seed in SEEDS) / len(SEEDS)
        for arm in ("targeted", "random", "seed_only")
    }
    return {
        "images_per_run": len(rows),
        "prediction_count": len(rows) * len(results),
        "runs": results,
        "mean_AP": means,
        "scope": "Saved predictions and hash-verified reference labels; no image/weight verification or neural execution",
        "current_evaluation_source_sha256": file_hash(ROOT / "src/driving_data/evaluation.py"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    result = replay(args.labels_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "result.json", result)
    print(
        f"Recomputed {result['prediction_count']} predictions; every saved metric matched. Mean AP: {result['mean_AP']}"
    )


if __name__ == "__main__":
    main()
