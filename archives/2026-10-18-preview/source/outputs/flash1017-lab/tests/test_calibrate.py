import pytest

from flash1017.calibrate import fit


def test_censored_suffix_is_not_labeled_as_rejected():
    runs = []
    for k in range(4):
        events = [{"copied": False, "remaining": 100, "verified": k,
                   "accepted": min(k, 1), "cycle_ms": 10 + k,
                   "confidence_logits": [1, 2, 99]} for _ in range(30)]
        runs.append({"type": "run", "profile": {"draft_width": 3, "forced_depth": k},
                     "events": events})
    result = fit(runs, width=3)
    assert result["cycle_ms"] == [10, 11, 12, 13]
    diagnostics = result["diagnostics"]["confidence"]
    assert diagnostics[0]["empirical_acceptance"] == 1
    assert diagnostics[1]["empirical_acceptance"] == 0
    assert diagnostics[2]["n"] == 0
    assert result["temperatures"][2] == 1


def test_missing_cost_depth_cannot_silently_produce_a_policy():
    with pytest.raises(ValueError):
        fit([], width=3)
