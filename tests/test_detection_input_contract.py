from copy import deepcopy

import pytest

from driving_data.evaluation import coco_metrics, match_objects


@pytest.fixture
def inputs():
    rows = [{"sample_id": "a", "width": 100, "height": 100}]
    labels = {"a": [{"box": [0, 0, 20, 20], "label": 1}]}
    preds = {"a": {"predictions": [{"box": [0, 0, 20, 20], "label": 1, "score": 0.9}]}}
    return rows, labels, preds


@pytest.mark.parametrize(
    "field,value",
    [
        ("score", float("nan")),
        ("score", float("inf")),
        ("score", 1.1),
        ("score", True),
        ("box", [20, 0, 0, 20]),
        ("box", [0, 0, float("inf"), 20]),
        ("label", 999),
        ("label", True),
    ],
)
def test_invalid_detection_rejected_by_both_metrics(inputs, field, value):
    rows, labels, preds = deepcopy(inputs)
    preds["a"]["predictions"][0][field] = value
    with pytest.raises(ValueError):
        coco_metrics(rows, labels, preds)
    with pytest.raises(ValueError):
        match_objects(labels["a"], preds["a"]["predictions"])


def test_duplicate_images_cannot_inflate_denominator(inputs):
    rows, labels, preds = inputs
    with pytest.raises(ValueError):
        coco_metrics(rows + rows, labels, preds)


def test_empty_detection_remains_a_miss(inputs):
    rows, labels, preds = inputs
    preds["a"]["predictions"] = []
    assert match_objects(labels["a"], [])["fn"] == 1
    assert coco_metrics(rows, labels, preds)["AP"] == 0.0
