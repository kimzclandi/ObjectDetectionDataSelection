"""Verify follow-up provenance and recompute its reported metrics; no new inference."""

from pathlib import Path

from driving_data.common import digest, read_json
from driving_data.evaluation import evaluate
from driving_data.vlm_quality import summarize

root = Path(__file__).resolve().parents[1]
manifest = read_json(root / "data/manifest.json")
by_id = {r["sample_id"]: r for r in manifest["rows"]}
controls = read_json(root / "reports/pilot/controls.json")
for row in controls["records"]:
    training = read_json(root / "reports/pilot" / row["name"] / "training.json")
    ids = training["recipe"]["image_ids"]
    assert len(ids) == 56 and len(set(ids)) == 44 and training["steps"] == 112
    assert all(by_id[sid]["split"] == "seed" for sid in ids)
    expected = read_json(root / "reports/pilot" / row["name"] / "evaluation.json")
    assert evaluate(root, row["name"]) == expected
    assert row["test_AP"] == expected["splits"]["test"]["coco"]["AP"]
for name in ("vlm_quality", "vlm_quality_constrained"):
    protocol = read_json(root / "reports/pilot" / name / "protocol.json")
    result = read_json(root / "reports/pilot" / name / "results.json")
    assert digest(protocol) == result["protocol_sha256"]
    assert len(result["rows"]) == 36
    assert all(
        by_id[r["sample_id"]]["split"] == "dev" and by_id[r["sample_id"]]["sha256"] == r["sha256"]
        for r in result["rows"]
    )
    assert all(
        summarize(result["rows"], field) == result["metrics"][field] for field in ("majority", "rule", "vlm")
    )
print("Verified 3 matched-step controls and recomputed both 36-image VLM evaluation reports")
