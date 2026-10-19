import math

import pytest
from hypothesis import given, strategies as st

from flash1017.policy import PrefixPolicy, Experiment


def test_stops_before_reading_future_confidence():
    def confidences():
        yield -10.0
        raise AssertionError("Decision used future candidate-dependent evidence")

    assert PrefixPolicy((10, 11, 12)).choose(confidences()).depth == 0


def test_never_jumps_over_first_slowdown_to_global_argmax():
    # An attractive later shape cannot justify peeking at later candidates.
    result = PrefixPolicy((10, 15, 6)).choose((0.0, 100.0))
    assert result.depth == 0


def test_expected_emission_includes_bonus_and_prefix_survival():
    result = PrefixPolicy((10, 11, 12)).choose((math.log(4), math.log(3)))
    assert result.depth == 2
    assert result.expected_tokens == pytest.approx(1 + .8 + .8 * .75)
    assert result.predicted_tokens_per_ms == pytest.approx(2.4 / 12)


@given(st.lists(st.floats(min_value=-100, max_value=100), min_size=1, max_size=9))
def test_unused_suffix_cannot_change_admitted_prefix(suffix):
    curve = tuple(10 + k for k in range(len(suffix) + 3))
    assert PrefixPolicy(curve).choose((10, -100, *suffix)).depth == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_invalid_confidence_keeps_last_causal_prefix(bad):
    assert PrefixPolicy((10, 11, 12)).choose((10, bad)).depth == 1


@pytest.mark.parametrize("costs", [(), (1,), (1, 0), (1, -1), (1, float("nan"))])
def test_bad_costs_rejected(costs):
    with pytest.raises(ValueError):
        PrefixPolicy(costs)


def test_fixed_profile_uses_same_draft_width_and_accepts_zero():
    experiment = Experiment(draft_width=5, forced_depth=0)
    assert experiment.select([100] * 5, 5) == 0
    assert Experiment().select([0] * 3, 3) == 3


def test_calibration_is_frozen_and_changes_only_confidence():
    raw = PrefixPolicy((10, 14), (1,)).choose((.2,))
    calibrated = PrefixPolicy((10, 14), (.1,)).choose((.2,))
    assert raw.depth == calibrated.depth == 1
    assert calibrated.expected_tokens > raw.expected_tokens
