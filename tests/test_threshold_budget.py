import numpy as np
import pytest

from telemetry_anomdet.thresholding import anomalous_sequences, threshold_for_budget


def test_returns_the_documented_keys():
    out = threshold_for_budget(np.arange(100.0), budget=0.1)
    assert set(out) == {"threshold", "flagged", "n_above", "n_sequences"}


def test_threshold_is_an_observed_value_not_an_interpolated_one():
    """
    The cutoff snaps up to a real sample. An interpolated cutoff falls between
    two observations and lets more than the budget sit above it.
    """
    errors = np.arange(1000.0)
    out = threshold_for_budget(errors, budget=0.05)
    assert out["threshold"] == pytest.approx(np.quantile(errors, 0.95, method="higher"))
    assert out["threshold"] in set(errors.tolist())


def test_flagged_fraction_tracks_the_budget():
    rng = np.random.default_rng(0)
    errors = rng.normal(size=10_000)
    for budget in (0.001, 0.01, 0.05, 0.25):
        out = threshold_for_budget(errors, budget=budget)
        assert out["flagged"] == pytest.approx(budget, abs=1e-3)


def test_counts_agree_with_the_threshold():
    rng = np.random.default_rng(1)
    errors = rng.normal(size=500)
    out = threshold_for_budget(errors, budget=0.1)

    above = errors > out["threshold"]
    assert out["n_above"] == int(above.sum())
    assert out["flagged"] == pytest.approx(above.mean())
    assert out["n_sequences"] == len(anomalous_sequences(errors, out["threshold"]))


def test_a_tighter_budget_never_lowers_the_threshold():
    rng = np.random.default_rng(2)
    errors = rng.normal(size=2000)
    thresholds = [
        threshold_for_budget(errors, budget=b)["threshold"] for b in (0.5, 0.2, 0.05, 0.01)
    ]
    assert thresholds == sorted(thresholds)


def test_accepts_a_plain_sequence():
    out = threshold_for_budget([0.0, 1.0, 2.0, 3.0, 100.0], budget=0.25)
    assert out["n_above"] == 1


def test_constant_signal_flags_nothing():
    """
    Every value equals the quantile, and the comparison is strict, so nothing
    is above it. Flagging nothing satisfies "no more than the budget"; a
    constant error signal carries no anomaly to find.
    """
    out = threshold_for_budget(np.ones(100), budget=0.1)
    assert out["n_above"] == 0
    assert out["flagged"] == 0.0
    assert out["n_sequences"] == 0


def test_sequences_are_counted_as_runs_not_points():
    # Two runs of two points each, well above the rest.
    errors = np.zeros(100)
    errors[10:12] = 50.0
    errors[60:62] = 50.0
    out = threshold_for_budget(errors, budget=0.05)
    assert out["n_above"] == 4
    assert out["n_sequences"] == 2


@pytest.mark.parametrize("budget", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_budget_outside_the_open_unit_interval_raises(budget):
    with pytest.raises(ValueError, match="budget must lie in"):
        threshold_for_budget(np.arange(10.0), budget=budget)


def test_empty_errors_raise():
    with pytest.raises(ValueError, match="must not be empty"):
        threshold_for_budget([], budget=0.1)


def test_flagged_stays_close_to_the_budget():
    """
    The flagged fraction is approximate, not bounded. It tracks the budget to
    within a point or two at realistic sizes, which is the property the
    docstring promises and the benchmark relies on.
    """
    rng = np.random.default_rng(3)
    for n in (200, 1000, 5000):
        errors = rng.normal(size=n)
        for budget in (0.01, 0.02, 0.05, 0.1, 0.25):
            out = threshold_for_budget(errors, budget=budget)
            assert out["flagged"] == pytest.approx(budget, abs=2.0 / n), f"n={n} budget={budget}"


def test_small_inputs_respect_the_budget():
    """
    The case that used to break the bound. Linear interpolation put the cutoff
    between two samples and flagged 3 of 50 (0.06) at budget 0.05, where 2
    (0.04) fits. Snapping the quantile up to a real sample holds the bound.
    """
    out = threshold_for_budget(np.arange(50.0), budget=0.05)
    assert out["n_above"] == 2
    assert out["flagged"] <= 0.05


def test_the_budget_is_a_bound_at_every_size():
    """
    Swept rather than spot-checked, because the failure only appeared at some
    combinations of length and budget.
    """
    rng = np.random.default_rng(4)
    for n in (7, 23, 50, 200, 1000):
        errors = rng.normal(size=n)
        for budget in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5):
            out = threshold_for_budget(errors, budget=budget)
            assert out["flagged"] <= budget + 1e-12, f"n={n} budget={budget}"
