import json
import sqlite3

import pytest

from driving_data import model
from driving_data.common import file_hash, write_json


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    image = tmp_path / "data/images/a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture bytes, predict_one mocked")
    row = {"sample_id": "a", "sha256": file_hash(image), "image_path": "data/images/a.jpg", "split": "dev"}
    write_json(tmp_path / "data/manifest.json", {"version": "v1", "rows": [row]})
    cfg = {"weight_sha256": "first-model"}
    calls = []
    monkeypatch.setattr(model, "load_model", lambda *a: (None, dict(cfg)))

    def predict(*args):
        calls.append(1)
        return dict(
            sample_id="a", sha256=row["sha256"], predictions=[], embedding=[1.0, 0.0], latency_seconds=0.01
        )

    monkeypatch.setattr(model, "predict_one", predict)
    return tmp_path, cfg, calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_id", "other"),
        ("sha256", "other"),
        ("embedding", [float("nan")]),
        ("latency_seconds", -1),
        ("predictions", [{"box": [0, 0, 2, 2], "label": 3, "score": float("nan")}]),
    ],
)
def test_corrupt_cached_result_is_rejected(runtime, field, value):
    root, cfg, calls = runtime
    model.infer(root, splits=("dev",))
    path = root / "reports/pilot/baseline/predictions.json"
    old = path.read_bytes()
    db = next((root / "work/jobs").glob("*.sqlite"))
    with sqlite3.connect(db) as con:
        r = json.loads(con.execute("SELECT result FROM jobs").fetchone()[0])
        r[field] = value
        con.execute("UPDATE jobs SET result=?", (json.dumps(r),))
    with pytest.raises(ValueError):
        model.infer(root, splits=("dev",))
    assert path.read_bytes() == old and len(calls) == 1


def test_changed_model_cannot_overwrite_named_run(runtime):
    root, cfg, calls = runtime
    model.infer(root, splits=("dev",))
    path = root / "reports/pilot/baseline/predictions.json"
    old = path.read_bytes()
    cfg["weight_sha256"] = "second-model"
    with pytest.raises(ValueError):
        model.infer(root, splits=("dev",))
    assert path.read_bytes() == old and len(calls) == 1
    assert model.infer(root, name="second", splits=("dev",))["computed"] == 1


def test_valid_resume_is_a_cache_hit(runtime):
    root, _, calls = runtime
    first = model.infer(root, splits=("dev",))
    second = model.infer(root, splits=("dev",))
    assert (
        first["computed"] == 1 and second["computed"] == 0 and second["cache_hits"] == 1 and len(calls) == 1
    )


def test_interrupted_job_is_recovered(runtime):
    root, _, calls = runtime
    with pytest.raises(SystemExit):
        model.infer(root, splits=("dev",), crash_after=0)
    assert not (root / "reports/pilot/baseline/predictions.json").exists()
    result = model.infer(root, splits=("dev",))
    assert result["recovered_jobs"] == 1 and result["computed"] == 1 and len(calls) == 1


def test_invalid_new_prediction_never_becomes_a_successful_job(runtime, monkeypatch):
    root, _, _ = runtime
    monkeypatch.setattr(
        model,
        "predict_one",
        lambda *args: dict(
            sample_id="a", sha256="wrong", predictions=[], embedding=[1.0], latency_seconds=0.1
        ),
    )
    with pytest.raises(ValueError):
        model.infer(root, splits=("dev",))
    db = next((root / "work/jobs").glob("*.sqlite"))
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT status,attempts FROM jobs").fetchone() == ("failed", 2)
    assert not (root / "reports/pilot/baseline/predictions.json").exists()
