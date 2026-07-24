# tests/test_evaluation.py

"""Unit tests for point-adjusted evaluation metrics."""

import numpy as np
import pytest

from telemetry_anomdet.evaluation import (
    best_point_adjusted_f1,
    point_adjust,
    point_adjusted_f1,
    prf,
    windows_to_point_scores,
    windows_to_points,
)

# --------------------------------------------------------------------------- #
# point_adjust
# --------------------------------------------------------------------------- #


def test_point_adjust_expands_hit_segment():
    truth = [False, True, True, True, False]
    pred = [False, False, True, False, False]  # one hit inside the segment
    adjusted = point_adjust(pred, truth)
    # the whole [1:4) segment becomes predicted
    assert adjusted.tolist() == [False, True, True, True, False]


def test_point_adjust_leaves_missed_segment_alone():
    truth = [False, True, True, False]
    pred = [False, False, False, False]  # no hit in the segment
    assert point_adjust(pred, truth).tolist() == [False, False, False, False]


def test_point_adjust_keeps_false_positives_outside_segments():
    truth = [False, True, True, False, False]
    pred = [True, False, False, True, False]  # FP at 0 and 3, no hit in segment
    adjusted = point_adjust(pred, truth)
    # segment [1:3) not hit -> stays 0; the FPs remain
    assert adjusted.tolist() == [True, False, False, True, False]


def test_point_adjust_handles_segment_at_end():
    truth = [False, False, True, True]
    pred = [False, False, False, True]
    assert point_adjust(pred, truth).tolist() == [False, False, True, True]


def test_point_adjust_shape_mismatch_raises():
    with pytest.raises(ValueError):
        point_adjust([True, False], [True, False, True])


# --------------------------------------------------------------------------- #
# prf / point_adjusted_f1
# --------------------------------------------------------------------------- #


def test_prf_basic():
    pred = [True, True, False, False]
    truth = [True, False, False, True]
    m = prf(pred, truth)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


def test_prf_all_zero_predictions():
    m = prf([False, False], [True, True])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_point_adjusted_f1_beats_raw_when_partial_hit():
    truth = [False, True, True, True, False]
    pred = [False, False, True, False, False]  # 1/3 of the segment hit

    raw = prf(pred, truth)
    adj = point_adjusted_f1(pred, truth)

    assert raw["recall"] < 1.0  # raw only catches 1 of 3 points
    assert adj["recall"] == 1.0  # adjustment credits the whole segment
    assert adj["f1"] >= raw["f1"]


# --------------------------------------------------------------------------- #
# windows_to_points
# --------------------------------------------------------------------------- #


def test_windows_to_points_expands_flags():
    # window_size=3, step=2. window 1 flagged -> points [2, 3, 4]
    flags = [False, True, False]
    pts = windows_to_points(flags, n_points=8, window_size=3, step=2)
    expected = [False, False, True, True, True, False, False, False]
    assert pts.tolist() == expected


def test_windows_to_points_overlapping_windows_union():
    # windows overlap; flagged windows 0 and 1 both contribute
    flags = [True, True]
    pts = windows_to_points(flags, n_points=6, window_size=3, step=2)
    # window0 -> [0,1,2], window1 -> [2,3,4]
    assert pts.tolist() == [True, True, True, True, True, False]


def test_windows_to_points_no_flags():
    pts = windows_to_points([False, False], n_points=5, window_size=2, step=2)
    assert not pts.any()
    assert len(pts) == 5


# --------------------------------------------------------------------------- #
# windows_to_point_scores
# --------------------------------------------------------------------------- #


def test_windows_to_point_scores_max_over_covering_windows():
    # window_size=3, step=2. scores [0.2, 0.9]
    # window0 covers [0,1,2] with 0.2; window1 covers [2,3,4] with 0.9
    pts = windows_to_point_scores([0.2, 0.9], n_points=6, window_size=3, step=2)
    # point 2 is covered by both -> max(0.2, 0.9) = 0.9
    assert pts[2] == 0.9
    assert pts[0] == 0.2 and pts[1] == 0.2
    assert pts[3] == 0.9 and pts[4] == 0.9
    # point 5 covered by no window -> falls back to min score
    assert pts[5] == 0.2


# --------------------------------------------------------------------------- #
# best_point_adjusted_f1
# --------------------------------------------------------------------------- #


def test_best_point_adjusted_f1_finds_separating_threshold():
    # scores cleanly separate anomaly (high) from normal (low)
    scores = np.array([0.1, 0.2, 0.9, 0.95, 0.15])
    truth = np.array([False, False, True, True, False])
    best = best_point_adjusted_f1(scores, truth)
    assert best["f1"] == 1.0
    assert 0.2 < best["threshold"] <= 0.9


def test_best_point_adjusted_f1_reports_threshold_and_beats_fixed():
    scores = np.array([0.1, 0.4, 0.5, 0.8, 0.9])
    truth = np.array([False, True, True, True, False])
    best = best_point_adjusted_f1(scores, truth)
    # a good threshold should recover the middle segment with point adjustment
    assert best["recall"] == 1.0
    assert "threshold" in best
