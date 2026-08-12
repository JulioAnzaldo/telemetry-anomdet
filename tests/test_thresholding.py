import numpy as np
import pytest

from telemetry_anomdet.thresholding import (
    anomalous_sequences,
    detect_anomalies,
    dynamic_threshold,
    filter_sequences,
    prune_sequences,
    startup_skip,
)

# ---------------------------------------------------------------------------
# Contiguous runs
# ---------------------------------------------------------------------------


def test_finds_runs_above_threshold():
    e = np.array([0.0, 5.0, 5.0, 0.0, 0.0, 7.0, 0.0])
    assert anomalous_sequences(e, 1.0) == [(1, 2), (5, 5)]


def test_runs_touching_either_end_are_closed():
    e = np.array([9.0, 9.0, 0.0, 9.0])
    assert anomalous_sequences(e, 1.0) == [(0, 1), (3, 3)]


def test_no_runs_when_nothing_exceeds():
    assert anomalous_sequences(np.array([1.0, 2.0]), 5.0) == []


def test_threshold_is_strict():
    """A value equal to the threshold is nominal, matching `> threshold`."""
    assert anomalous_sequences(np.array([1.0, 2.0, 1.0]), 2.0) == []


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def _signal_with_spike(n=500, spike_at=250, spike=12.0, seed=0):
    rng = np.random.default_rng(seed)
    e = np.abs(rng.normal(0.0, 1.0, n))
    e[spike_at : spike_at + 5] = spike
    return e


def test_isolates_an_obvious_spike():
    e = _signal_with_spike()
    got = dynamic_threshold(e)
    assert got["n_sequences"] == 1
    assert got["n_above"] == 5
    # The threshold must sit between the nominal bulk and the spike.
    assert e[e < 10].max() < got["threshold"] < 12.0


def test_flat_signal_flags_nothing():
    got = dynamic_threshold(np.full(100, 3.0))
    assert got["n_above"] == 0
    assert got["threshold"] == 3.0


def test_pure_noise_flags_little_or_nothing():
    """Without a real outlier, the cost terms should suppress the selection."""
    e = np.abs(np.random.default_rng(1).normal(0.0, 1.0, 2000))
    got = dynamic_threshold(e)
    assert got["n_above"] <= 0.01 * e.size


def test_fragmentation_is_penalised():
    """
    Scattered exceedances cost more than the same count in one run.

    The |E_seq|**2 term is what makes a single coherent event preferred, so the
    objective must score the clustered signal higher.
    """
    n = 400
    rng = np.random.default_rng(2)
    base = np.abs(rng.normal(0.0, 1.0, n))

    clustered = base.copy()
    clustered[100:110] = 9.0

    scattered = base.copy()
    scattered[rng.choice(n, 10, replace=False)] = 9.0

    assert dynamic_threshold(clustered)["score"] > dynamic_threshold(scattered)["score"]


def test_rejects_empty_errors():
    with pytest.raises(ValueError, match="must not be empty"):
        dynamic_threshold(np.array([]))


def test_uses_no_labels():
    """The selection depends only on the errors, so identical input agrees."""
    e = _signal_with_spike()
    assert dynamic_threshold(e) == dynamic_threshold(e.copy())


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def test_prunes_a_sequence_close_to_the_nominal_tail():
    e = np.zeros(60)
    e[:] = 1.0
    e[10:12] = 20.0  # clearly anomalous
    e[30:32] = 1.05  # barely above the nominal level
    seqs = anomalous_sequences(e, 1.02)
    assert len(seqs) == 2
    kept = prune_sequences(e, seqs, 1.02, min_decrease=0.13)
    assert kept == [(10, 11)]


def test_keeps_sequences_that_decline_gradually():
    e = np.ones(80) * 0.1
    e[10:12] = 10.0
    e[30:32] = 9.5
    e[50:52] = 9.0
    seqs = anomalous_sequences(e, 1.0)
    assert prune_sequences(e, seqs, 1.0, min_decrease=0.13) == seqs


def test_pruning_preserves_input_order():
    e = np.ones(60) * 0.1
    e[40:42] = 10.0  # later in time, larger peak
    e[10:12] = 9.5
    seqs = anomalous_sequences(e, 1.0)
    kept = prune_sequences(e, seqs, 1.0, min_decrease=0.13)
    assert kept == sorted(kept)


def test_pruning_empty_input():
    assert prune_sequences(np.array([1.0]), [], 0.5) == []


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_detect_marks_only_the_spike():
    e = _signal_with_spike()
    got = detect_anomalies(e)
    assert got["sequences"] == [(250, 254)]
    assert got["mask"].sum() == 5
    assert got["mask"][250:255].all()


def test_detect_accepts_a_supplied_threshold():
    """An operating point fitted on training errors can be applied directly."""
    e = _signal_with_spike()
    got = detect_anomalies(e, threshold=6.0)
    assert got["threshold"] == 6.0
    assert got["sequences"] == [(250, 254)]


def test_detect_can_skip_pruning():
    e = np.ones(60) * 1.0
    e[10:12] = 20.0
    e[30:32] = 1.05
    unpruned = detect_anomalies(e, threshold=1.02, prune=False)
    pruned = detect_anomalies(e, threshold=1.02, prune=True)
    assert len(unpruned["sequences"]) == 2
    assert len(pruned["sequences"]) == 1
    assert pruned["n_pruned"] == 1


def test_detect_on_a_clean_signal_flags_nothing():
    got = detect_anomalies(np.full(200, 2.0))
    assert not got["mask"].any()
    assert got["sequences"] == []


# ---------------------------------------------------------------------------
# Candidate placement
# ---------------------------------------------------------------------------


def test_both_strategies_isolate_a_clean_spike():
    e = _signal_with_spike()
    for strategy in ("sigma", "quantile"):
        got = dynamic_threshold(e, strategy=strategy)
        assert got["strategy"] == strategy
        assert got["n_sequences"] == 1, strategy


def test_strategies_can_select_different_thresholds():
    """
    The two candidate sets are placed differently, so they need not agree.

    Which is better is a property of the data rather than something to assert
    here; see the anomaly scoring page of the documentation.
    """
    rng = np.random.default_rng(7)
    e = rng.lognormal(0.0, 1.6, 1500)
    e[600:612] = np.percentile(e, 99.0) * 1.4

    quantile = dynamic_threshold(e, strategy="quantile")["threshold"]
    sigma = dynamic_threshold(e, strategy="sigma")["threshold"]
    assert quantile != sigma


def test_objective_prefers_a_cheap_flag_over_a_costly_run():
    """
    Documents a real bias in the criterion, and the reason recall runs low.

    Cost is `n_above + n_sequences ** 2`, so flagging one point costs 2 while a
    twelve point run costs 13. When a single extreme value already accounts for
    most of the reduction in mean and standard deviation, the objective takes it
    and leaves the run alone.
    """
    rng = np.random.default_rng(4)
    e = np.abs(rng.normal(0.0, 1.0, 1000))
    e[500:505] = 40.0  # a contiguous, genuinely anomalous run
    e[700] = 4000.0  # one enormous isolated value

    got = dynamic_threshold(e)
    assert got["n_above"] == 1
    assert got["n_sequences"] == 1
    # The isolated value is taken; the run is not.
    assert e[700] > got["threshold"] >= 40.0


def test_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        dynamic_threshold(np.array([1.0, 2.0, 3.0]), strategy="magic")


def test_unknown_strategy_rejected_even_on_a_flat_signal():
    """The early return for a flat signal must not skip validation."""
    with pytest.raises(ValueError, match="strategy"):
        dynamic_threshold(np.full(50, 2.0), strategy="magic")


# ---------------------------------------------------------------------------
# Protocol filters
# ---------------------------------------------------------------------------


def test_filter_drops_single_sample_runs():
    """A lone sample above threshold is a spike, not an event."""
    seqs = [(10, 10), (20, 25), (30, 30)]
    assert filter_sequences(seqs, min_length=2) == [(20, 25)]


def test_filter_keeps_everything_at_min_length_one():
    seqs = [(10, 10), (20, 25)]
    assert filter_sequences(seqs, min_length=1) == seqs


def test_filter_drops_sequences_in_the_cold_start():
    """A sequence ending before the cutoff is discarded; one straddling it is not."""
    seqs = [(0, 40), (45, 60), (100, 120)]
    assert filter_sequences(seqs, min_length=1, ignore_before=50) == [(45, 60), (100, 120)]


def test_startup_skip_follows_the_published_rule():
    assert startup_skip(5000, 250) == 500
    assert startup_skip(2000, 250) == 250
    assert startup_skip(1000, 250) == 0


def test_startup_skip_boundaries():
    assert startup_skip(2500, 250) == 500
    assert startup_skip(2499, 250) == 250
    assert startup_skip(1800, 250) == 250
    assert startup_skip(1799, 250) == 0
