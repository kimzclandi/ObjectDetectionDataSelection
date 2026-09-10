import copy

import pytest

from driving_data.common import digest, image_path
from driving_data.data import convert_labels, split_for, validate_partition
from driving_data.evaluation import coco_metrics, match_objects
from driving_data.selection import select
from driving_data.store import Store, runner_lock
from driving_data.training import regression_gate


def row(sid="a", split="pool", sha="hash"):
    return {
        "sample_id": sid,
        "group_id": sid,
        "split": split,
        "image_path": f"data/images/{sid}.jpg",
        "sha256": sha,
    }


def pred(sid="a", sha="hash", score=0.5, embedding=None):
    return {
        "sample_id": sid,
        "sha256": sha,
        "predictions": [{"box": [0, 0, 10, 10], "score": score, "label": 3}],
        "embedding": embedding or [1.0, 0.0],
    }


def test_digest_key_order_and_nan_rejection():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        digest(float("nan"))


def test_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        image_path(tmp_path, {"image_path": "../secret"})


def test_same_group_stays_in_same_split():
    assert split_for("b1c81faa") == split_for("b1c81faa")
    rows = [row("a", "seed"), row("b", "test", "other")]
    rows[1]["group_id"] = "a"
    with pytest.raises(ValueError, match="Group leakage"):
        validate_partition(rows)


def test_duplicates_rejected_across_splits():
    with pytest.raises(ValueError, match="Duplicate image"):
        validate_partition([row("a", "seed"), row("b", "test")])


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="Duplicate sample"):
        validate_partition([row(), row(sha="other")])


def test_class_mapping_and_clipping():
    sample = {
        "metadata": {"width": 100, "height": 50},
        "detections": {
            "detections": [
                {"label": "rider", "bounding_box": [-0.01, 0.2, 0.3, 0.4]},
                {"label": "traffic sign", "bounding_box": [0, 0, 1, 1]},
            ]
        },
    }
    labels = convert_labels(sample)
    assert len(labels) == 1 and labels[0]["label"] == 1
    assert labels[0]["box"] == pytest.approx([0, 10, 29, 30])


@pytest.mark.parametrize("box", [[0, 0, -1, 1], [0, 0, float("nan"), 1], [2, 0, 1, 1]])
def test_invalid_boxes_rejected(box):
    with pytest.raises(ValueError):
        convert_labels(
            {
                "metadata": {"width": 100, "height": 50},
                "detections": {"detections": [{"label": "car", "bounding_box": box}]},
            }
        )


def test_one_gt_cannot_match_twice():
    gold = [{"box": [0, 0, 10, 10], "label": 3}]
    p = pred()["predictions"][0]
    m = match_objects(gold, [p, p])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0)


def test_wrong_class_does_not_match():
    m = match_objects([{"box": [0, 0, 10, 10], "label": 1}], pred()["predictions"])
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 1)


def test_below_operating_threshold_is_not_false_positive():
    m = match_objects([], pred(score=0.49)["predictions"])
    assert m["fp"] == 0


def test_coco_perfect_and_no_detection():
    rows = [{"sample_id": "a", "width": 100, "height": 100}]
    gold = {"a": [{"box": [0, 0, 10, 10], "label": 3}]}
    assert coco_metrics(rows, gold, {"a": pred()})["AP"] == pytest.approx(1.0)
    assert coco_metrics(rows, gold, {"a": {"predictions": []}})["AP"] == 0.0


@pytest.mark.parametrize("field", ["labels", "ground_truth", "timeofday", "weather"])
def test_sampler_rejects_privileged_fields(field):
    r = row()
    r[field] = "secret"
    with pytest.raises(ValueError, match="GT forbidden"):
        select([r], [pred()], 1, "random", 42)


def test_sampler_cannot_read_test_rows():
    with pytest.raises(ValueError):
        select([row(split="test")], [pred()], 1, "random", 42)


@pytest.mark.parametrize("budget", [0, -1, 2])
def test_sampler_budget(budget):
    with pytest.raises(ValueError):
        select([row()], [pred()], budget, "random", 42)


def test_sampler_rejects_stale_predictions():
    with pytest.raises(ValueError, match="stale"):
        select([row()], [pred(sha="different")], 1, "random", 42)


def test_selection_deterministic_and_diverse():
    pool = [row("a"), row("b", sha="b"), row("c", sha="c")]
    predictions = [pred(), pred("b", "b", embedding=[1.0, 0.0]), pred("c", "c", embedding=[0.0, 1.0])]
    first = select(pool, predictions, 2, "diverse", 42)
    assert first == select(list(reversed(pool)), predictions, 2, "diverse", 42)
    assert "c" in {r["sample_id"] for r in first}
    assert len({r["sample_id"] for r in first}) == 2


def test_recovery_and_idempotent_commit(tmp_path):
    path = tmp_path / "jobs.sqlite"
    s = Store(path)
    s.enqueue("k", "a")
    s.start("k")
    s.close()
    s = Store(path)
    assert s.recover() == 1
    s.start("k")
    s.succeed("k", {"answer": 1})
    s.enqueue("k", "a")
    assert s.get("k") == {"answer": 1}
    assert s.counts() == {"succeeded": 1}
    assert s.db.execute("SELECT attempts FROM jobs").fetchone()[0] == 2
    s.close()


def test_failed_job_is_retryable(tmp_path):
    s = Store(tmp_path / "jobs.sqlite")
    s.enqueue("k", "a")
    s.start("k")
    s.fail("k", "OSError")
    assert s.get("k") is None
    s.start("k")
    s.succeed("k", {"ok": True})
    assert s.counts() == {"succeeded": 1}
    s.close()


def test_only_one_runner(tmp_path):
    path = tmp_path / "jobs.sqlite"
    with runner_lock(path):
        with pytest.raises(BlockingIOError):
            with runner_lock(path):
                pass


def test_gate_insufficient_support_is_not_pass():
    b = {
        "splits": {
            "test": {
                "coco": {"AP": 0.2},
                "slices": {"timeofday:night": {"images": 5, "objects": 30, "recall": 0.4}},
            }
        }
    }
    c = copy.deepcopy(b)
    c["splits"]["test"]["coco"]["AP"] = 0.3
    gate = {
        "max_overall_AP_drop": 0.01,
        "max_night_recall_drop": 0.05,
        "minimum_night_images": 20,
        "minimum_night_objects": 100,
    }
    assert regression_gate(b, c, gate)["decision"] == "INSUFFICIENT_EVIDENCE"
    c["splits"]["test"]["coco"]["AP"] = 0.1
    assert regression_gate(b, c, gate)["decision"] == "REJECT"
