import copy

import numpy as np
import pytest

from driving_data.failure_loop import near_duplicate, rank_pool, schedule, select


def test_selector_cannot_access_oracle_and_is_reproducible():
    pool = [
        dict(sample_id=str(i), group_id=str(i), split="pool", image_path=str(i), sha256=str(i))
        for i in range(50)
    ]
    predictions = {
        str(i): dict(sha256=str(i), embedding=[1.0, 0.0], predictions=[dict(score=0.2)]) for i in range(50)
    }
    ranks = rank_pool(pool, predictions, [1.0, 0.0])
    assert select(ranks, "targeted", 101) == select(ranks, "targeted", 101)
    assert len(select(ranks, "random", 101)) == 12
    assert {r["sample_id"] for r in select(ranks, "targeted", 101)} <= {r["sample_id"] for r in ranks[:36]}
    bad = copy.deepcopy(pool)
    bad[0]["labels"] = []
    with pytest.raises(ValueError, match="Unlabeled"):
        rank_pool(bad, predictions, [1.0, 0.0])
    bad = copy.deepcopy(predictions)
    bad["0"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        rank_pool(pool, bad, [1.0, 0.0])


def test_matched_steps_and_no_test_in_training():
    seed = [dict(sample_id=f"s{i}", split="seed") for i in range(44)]
    pool = [dict(sample_id=f"p{i}", split="pool") for i in range(12)]
    targeted = schedule(seed, pool, 101)
    control = schedule(seed, [], 101)
    assert len(targeted) == len(control) == 112
    assert len({r["sample_id"] for r in targeted}) == 56
    assert len({r["sample_id"] for r in control}) == 44
    pool[0]["split"] = "test"
    with pytest.raises(ValueError, match="leaked"):
        schedule(seed, pool, 101)


def test_duplicate_requires_hash_and_brightness_agreement():
    a = (np.zeros((16, 16)), np.zeros((16, 16), dtype=bool))
    assert near_duplicate(a, a)
    b = (np.ones((16, 16)) * 100, a[1])
    assert not near_duplicate(a, b)
