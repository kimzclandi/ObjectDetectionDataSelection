import pytest

from driving_data.common import digest, write_json
from driving_data.evaluation import evaluate


def setup_evaluation(root):
    gold = [{"box": [0, 0, 10, 10], "label": 3, "occluded": False, "truncated": False}]
    row = {
        "sample_id": "a",
        "split": "test",
        "group_id": "a",
        "sha256": "imagehash",
        "labels_sha256": digest(gold),
        "width": 100,
        "height": 100,
        "timeofday": "night",
        "weather": "clear",
        "scene": "street",
    }
    write_json(root / "data/manifest.json", {"version": "v1", "rows": [row]})
    write_json(root / "data/labels/a.json", gold)
    payload = {
        "dataset_version": "v1",
        "run_id": "model-v1",
        "rows": [{"sample_id": "a", "sha256": "imagehash", "predictions": []}],
    }
    write_json(root / "reports/pilot/baseline/predictions.json", payload)
    return payload


def test_empty_predictions_count_as_misses(tmp_path):
    setup_evaluation(tmp_path)
    result = evaluate(tmp_path, splits=("test",))
    assert result["splits"]["test"]["slices"]["all"]["fn"] == 1
    assert result["splits"]["test"]["coco"]["AP"] == 0


def test_missing_prediction_cannot_disappear_from_denominator(tmp_path):
    payload = setup_evaluation(tmp_path)
    payload["rows"] = []
    write_json(tmp_path / "reports/pilot/baseline/predictions.json", payload)
    with pytest.raises(ValueError, match="Missing or stale"):
        evaluate(tmp_path, splits=("test",))


def test_changed_eval_labels_are_rejected(tmp_path):
    setup_evaluation(tmp_path)
    write_json(tmp_path / "data/labels/a.json", [])
    with pytest.raises(ValueError, match="labels changed"):
        evaluate(tmp_path, splits=("test",))


def test_wrong_dataset_version_rejected(tmp_path):
    payload = setup_evaluation(tmp_path)
    payload["dataset_version"] = "v2"
    write_json(tmp_path / "reports/pilot/baseline/predictions.json", payload)
    with pytest.raises(ValueError, match="version mismatch"):
        evaluate(tmp_path, splits=("test",))
