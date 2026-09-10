"""Recompute metrics and sampling, repeat real CPU training and uncached inference.

Run from repository root after `driving-data experiment`.
"""

from pathlib import Path

import numpy as np
import torch

from driving_data.common import read_json, write_json
from driving_data.data import verify
from driving_data.evaluation import evaluate
from driving_data.model import load_model, predict_one
from driving_data.selection import run_selection
from driving_data.training import train

ROOT = Path(__file__).resolve().parents[1]
report = {"dataset": verify(ROOT)}
manifest = read_json(ROOT / "data/manifest.json")
config = read_json(ROOT / "configs/pilot.json")
old_selections = read_json(ROOT / "reports/pilot/selections.json")
run_selection(ROOT, config["budget_images"], tuple(config["strategy_seeds"]))
assert old_selections == read_json(ROOT / "reports/pilot/selections.json")
report["identical_selection_plans"] = len(old_selections["plans"])
experiment = read_json(ROOT / "reports/pilot/experiment.json")
for name in ["baseline"] + [r["name"] for r in experiment["records"]]:
    expected = read_json(ROOT / "reports/pilot" / name / "evaluation.json")
    assert evaluate(ROOT, name) == expected, name
report["identical_recomputed_evaluations"] = 10
original = torch.load(ROOT / "models/warmup.pth", map_location="cpu", weights_only=True)
pretrained_model, _ = load_model(ROOT, None, "cpu")
pretrained = pretrained_model.state_dict()  # apply Torchvision's legacy checkpoint key migration
frozen = [k for k in original if not k.startswith("roi_heads.box_predictor.")]
assert all(torch.equal(original[k], pretrained[k]) for k in frozen)
assert any(not torch.equal(original[k], pretrained[k]) for k in original if k not in frozen)
report["frozen_tensors_unchanged"] = len(frozen)
report["predictor_actually_updated"] = True
seed_rows = [r for r in manifest["rows"] if r["split"] == "seed"]
# Force a new training receipt on each verification; checkpoint is an isolated scratch output.
repro_receipt = ROOT / "reports/pilot/repro-warmup/training.json"
repro_receipt.unlink(missing_ok=True)
repeated = train(ROOT, "repro-warmup", seed_rows, config["warmup_seed"], config["warmup_epochs"], config)
new = torch.load(repeated, map_location="cpu", weights_only=True)
assert original.keys() == new.keys()
assert all(torch.equal(original[k], new[k]) for k in original)
report["repeated_training_identical_state_tensors"] = len(original)
model, _ = load_model(ROOT, ROOT / "models/warmup.pth", "cpu")
saved = {r["sample_id"]: r for r in read_json(ROOT / "reports/pilot/baseline/predictions.json")["rows"]}
max_difference = 0.0
for row in manifest["rows"][:12]:
    actual = predict_one(model, ROOT, row, "cpu")
    expected = saved[row["sample_id"]]
    assert len(actual["predictions"]) == len(expected["predictions"])
    for a, b in zip(actual["predictions"], expected["predictions"]):
        assert a["label"] == b["label"]
        delta = float(
            np.max(np.abs(np.asarray(a["box"] + [a["score"]]) - np.asarray(b["box"] + [b["score"]])))
        )
        max_difference = max(max_difference, delta)
    np.testing.assert_allclose(actual["embedding"], expected["embedding"], rtol=1e-6, atol=1e-6)
assert max_difference <= 1e-5
report["uncached_inference_images"] = 12
report["maximum_box_score_difference"] = max_difference
report["scope"] = (
    "Same-host CPU rerun; full warmup training, 12-image fresh inference, all saved evaluations and selections. Not a full nine-candidate independent rerun or cross-platform guarantee."
)
report["status"] = "passed"
write_json(ROOT / "reports/pilot/reproducibility.json", report)
print(report)
