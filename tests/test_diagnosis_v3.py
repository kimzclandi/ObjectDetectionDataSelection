import importlib.util
from pathlib import Path

import pytest

from driving_data.common import digest

spec = importlib.util.spec_from_file_location(
    "diagnosis_v3", Path(__file__).parents[1] / "scripts/diagnose_failure_v3.py"
)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def test_auc_ties_and_degenerate_labels():
    assert d.auc([True, False], [1, 1]) == 0.5
    assert d.auc([True, False], [0, 1]) == 0
    assert d.auc([True], [1]) is None
    assert d.ranks_corr([1, 1], [1, 2]) is None


def test_reject_evaluation_and_reference_drift():
    with pytest.raises(ValueError, match="development"):
        d.diagnosis([{"split": "test"}], {}, {})
    with pytest.raises(ValueError, match="identity"):
        d.diagnosis(
            [{"split": "dev", "sample_id": "x", "sha256": "a", "labels_sha256": "bad"}],
            {"x": {"sha256": "a"}},
            {"x": []},
        )


def test_group_exclusion_and_one_to_one_duplicate_fp():
    gold = [{"label": 1, "box": [0, 0, 10, 10]}]
    rows = [
        dict(sample_id=str(i), group_id=str(i // 2), split="dev", sha256="x", labels_sha256=digest(gold))
        for i in range(4)
    ]
    predictions = {
        r["sample_id"]: dict(
            sha256="x",
            embedding=[1.0, 0.0],
            predictions=[
                dict(label=1, box=[0, 0, 10, 10], score=0.3),
                dict(label=1, box=[0, 0, 10, 10], score=0.2),
            ],
        )
        for r in rows
    }
    result = d.diagnosis(rows, predictions, {r["sample_id"]: gold for r in rows})
    for r in result:
        assert r["group_id"] not in r["prototype_groups"]
        assert r["recoverable"] == 1
        assert r["low_fp"] == 1


def test_immutable_rejects_overwrite(tmp_path):
    path = tmp_path / "evidence.json"
    d.immutable(path, {"n": 1})
    d.immutable(path, {"n": 1})
    with pytest.raises(ValueError, match="drift"):
        d.immutable(path, {"n": 2})
