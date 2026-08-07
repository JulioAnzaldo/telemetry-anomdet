import numpy as np
import pytest

from telemetry_anomdet.evaluation import (
    best_point_adjusted_f1,
    false_alarm_rate_at_recall,
    pr_auc,
)

# ---------------------------------------------------------------------------
# PR-AUC
# ---------------------------------------------------------------------------


def test_perfect_ranking_scores_one():
    truth = np.array([False, False, True, True])
    assert pr_auc(np.array([0.1, 0.2, 0.9, 0.8]), truth) == pytest.approx(1.0)


def test_random_scores_land_on_the_base_rate():
    """The property that makes PR-AUC usable: random sits at the base rate."""
    rng = np.random.default_rng(0)
    n, rate = 20000, 0.13
    truth = rng.random(n) < rate
    got = pr_auc(rng.random(n), truth)
    assert got == pytest.approx(truth.mean(), abs=0.02)


def test_inverted_ranking_scores_below_base_rate():
    rng = np.random.default_rng(1)
    truth = rng.random(5000) < 0.2
    scores = np.where(truth, 0.0, 1.0) + rng.normal(0, 0.01, 5000)
    assert pr_auc(scores, truth) < truth.mean()


def test_degenerate_labels():
    assert pr_auc(np.array([0.5, 0.2]), np.array([False, False])) == 0.0
    assert pr_auc(np.array([0.5, 0.2]), np.array([True, True])) == 1.0


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must match"):
        pr_auc(np.zeros(3), np.zeros(4, dtype=bool))


def test_random_beats_a_real_detector_under_point_adjusted_f1():
    """
    Point adjustment rewards touching a segment, so random scores score highly
    on long segments while PR-AUC separates them correctly.
    """
    rng = np.random.default_rng(2)
    n = 6000
    truth = np.zeros(n, dtype=bool)
    for start in range(500, n, 1200):
        truth[start : start + 400] = True  # long contiguous segments

    random_scores = rng.random(n)
    assert best_point_adjusted_f1(random_scores, truth)["f1"] > 0.7
    assert pr_auc(random_scores, truth) == pytest.approx(truth.mean(), abs=0.03)


# ---------------------------------------------------------------------------
# False alarm rate at a stated recall
# ---------------------------------------------------------------------------


def test_perfect_detector_has_no_false_alarms():
    truth = np.array([False] * 90 + [True] * 10)
    scores = truth.astype(float)
    got = false_alarm_rate_at_recall(scores, truth, target_recall=1.0)
    assert got["recall"] == pytest.approx(1.0)
    assert got["false_alarm_rate"] == pytest.approx(0.0)


def test_random_detector_pays_the_target_recall_in_false_alarms():
    rng = np.random.default_rng(3)
    truth = rng.random(20000) < 0.1
    got = false_alarm_rate_at_recall(rng.random(20000), truth, target_recall=0.8)
    # With uninformative scores, catching 80% of anomalies means firing on
    # roughly 80% of nominal points too.
    assert got["false_alarm_rate"] == pytest.approx(0.8, abs=0.05)


def test_rejects_invalid_target_recall():
    truth = np.array([True, False])
    with pytest.raises(ValueError, match="target_recall"):
        false_alarm_rate_at_recall(np.array([1.0, 0.0]), truth, target_recall=0.0)
