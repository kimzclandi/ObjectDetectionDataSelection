"""Independent raw-prediction metric/provenance replay; no neural inference or training."""

from pathlib import Path

from driving_data.common import digest, file_hash, read_json
from driving_data.failure_loop import OUT, SEEDS, metrics, validate

protocol, manifest = validate()
rows = [r for r in manifest["rows"] if r["split"] == "test"]
labels = {r["sample_id"]: read_json(Path(f"data/labels/{r['sample_id']}.json")) for r in rows}
byid = {r["sample_id"]: r for r in rows}
allpreds = {}
for name in ["baseline"] + [
    f"{arm}-s{seed}" for seed in SEEDS for arm in ("targeted", "random", "seed_only")
]:
    files = sorted((OUT / "runs" / name / "predictions").glob("*.json"))
    assert len(files) == len(rows)
    preds = {}
    for f in files:
        record = read_json(f)
        pred = record["prediction"]
        r = byid[pred["sample_id"]]
        assert pred["sample_id"] not in preds
        assert pred["sha256"] == r["sha256"]
        assert record["key"] == digest([digest(protocol), record["model"], r["sample_id"], r["sha256"]])
        if name == "baseline":
            assert record["model"]["weight_sha256"] == protocol["initial_weights"]
        else:
            receipt = read_json(OUT / "runs" / name / "training.json")
            assert receipt["steps"] == 112
            assert receipt["initial_weights"] == protocol["initial_weights"]
            assert record["model"]["weight_sha256"] == receipt["checkpoint_sha256"]
        preds[pred["sample_id"]] = pred
    allpreds[name] = preds
    computed = metrics(rows, labels, preds, allpreds["baseline"])
    assert computed == read_json(OUT / "runs" / name / "evaluation.json"), name
summary = read_json(OUT / "summary.json")
for r in summary["records"]:
    assert r["AP"] == read_json(OUT / "runs" / r["name"] / "evaluation.json")["coco"]["AP"]
assert summary["mean_AP_delta"] == sum(summary["paired_AP_deltas"]) / 3
if (OUT / "artifact_manifest.json").exists():
    for p, sha in read_json(OUT / "artifact_manifest.json").items():
        assert file_hash(Path(p)) == sha, p
print(
    "Verified 10 x 120 raw predictions, every AP/recall/precision, 9 training receipts, fixed denominator and identity"
)
