import pytest

from driving_data.controls import matched_step_rows
from driving_data.vlm_quality import allowed_next, parse_label, summarize


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Daytime.", "daytime"),
        (" night ", "night"),
        ("daytime or night", None),
        ("not night", None),
        ("", None),
    ],
)
def test_vlm_strict_parser(raw, expected):
    assert parse_label(raw) == expected


def test_invalid_vlm_outputs_remain_in_denominator():
    result = summarize(
        [{"reference": "night", "vlm": None}, {"reference": "daytime", "vlm": "daytime"}], "vlm"
    )
    assert result["accuracy"] == 0.5
    assert result["invalid_outputs"] == 1
    assert result["classes"]["night"]["recall"] == 0


def test_constrained_prefix_trie_terminates_and_rejects_other_tokens():
    labels = [[10, 11], [20]]
    assert allowed_next([], labels, 99) == [10, 20]
    assert allowed_next([10], labels, 99) == [11]
    assert allowed_next([20], labels, 99) == [99]
    with pytest.raises(ValueError):
        allowed_next([30], labels, 99)


def test_control_repeats_seed_without_new_images():
    seed = [{"sample_id": str(i), "split": "seed"} for i in range(44)]
    rows = matched_step_rows(seed, 12, 17)
    assert len(rows) == 56
    assert len({r["sample_id"] for r in rows}) == 44
    assert rows == matched_step_rows(seed, 12, 17)
    with pytest.raises(ValueError):
        matched_step_rows([{"sample_id": "test", "split": "test"}], 0, 17)
