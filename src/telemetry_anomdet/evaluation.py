# src/telemetry_anomdet/evaluation.py

"""
Point-adjusted evaluation metrics for time-series anomaly detection.

Point adjustment is the standard SMAP / MSL protocol (Xu et al., 2018): if a
detector flags any point inside a labeled anomaly segment, the whole segment
counts as detected. It rewards catching an event even when not every point of
it is flagged, and makes results comparable to the telemanom and MEMTO
baselines that report point-adjusted F1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def point_adjust(pred: Sequence[bool], truth: Sequence[bool]) -> np.ndarray:
    """
    Apply point adjustment to point-level predictions.

    For each contiguous anomaly segment in ``truth``, if ``pred`` flags any point
    inside it, the entire segment is marked as predicted.

    Arguments:
        pred: Point-level boolean predictions.
        truth: Point-level boolean ground truth, same length as ``pred``.
    Returns:
        np.ndarray: The adjusted boolean prediction array.
    """

    pred = np.asarray(pred, dtype=bool).copy()
    truth = np.asarray(truth, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"pred and truth must match: {pred.shape} vs {truth.shape}")

    n = len(truth)
    i = 0
    while i < n:
        if not truth[i]:
            i += 1
            continue
        j = i
        while j < n and truth[j]:
            j += 1
        if pred[i:j].any():
            pred[i:j] = True
        i = j
    return pred


def prf(pred: Sequence[bool], truth: Sequence[bool]) -> dict:
    """
    Precision, recall, and F1 for point-level boolean arrays.

    Returns:
        dict: {'precision', 'recall', 'f1', 'tp', 'fp', 'fn'}.
    """

    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def point_adjusted_f1(pred: Sequence[bool], truth: Sequence[bool]) -> dict:
    """
    Point-adjusted precision, recall, and F1 (the standard SMAP metric).

    Equivalent to :func:`prf` computed on :func:`point_adjust` output.
    """

    return prf(point_adjust(pred, truth), truth)


def windows_to_points(
    window_flags: Sequence[bool], n_points: int, *, window_size: int, step: int
) -> np.ndarray:
    """
    Expand window-level flags to a point-level mask.

    Point ``t`` is flagged if it falls inside any flagged window. Window ``i``
    spans ``[i*step, i*step + window_size)``, matching ``windowify``.

    Arguments:
        window_flags: Boolean flag per window.
        n_points: Length of the point-level series the windows came from.
        window_size: Samples per window.
        step: Stride between windows.
    Returns:
        np.ndarray: Boolean mask of length ``n_points``.
    """

    window_flags = np.asarray(window_flags, dtype=bool)
    points = np.zeros(int(n_points), dtype=bool)
    for i, flag in enumerate(window_flags):
        if flag:
            s = i * step
            points[s : s + window_size] = True
    return points


def windows_to_point_scores(
    window_scores: Sequence[float], n_points: int, *, window_size: int, step: int
) -> np.ndarray:
    """
    Expand window-level scores to a point-level score array.

    Each point takes the maximum score among the windows covering it (window
    ``i`` spans ``[i*step, i*step + window_size)``). Points not covered by any
    window fall back to the minimum window score.

    Arguments:
        window_scores: Anomaly score per window (higher = more anomalous).
        n_points: Length of the point-level series.
        window_size: Samples per window.
        step: Stride between windows.
    Returns:
        np.ndarray: Float score array of length ``n_points``.
    """
    window_scores = np.asarray(window_scores, dtype=float)
    points = np.full(int(n_points), -np.inf)
    for i, s in enumerate(window_scores):
        a = i * step
        points[a : a + window_size] = np.maximum(points[a : a + window_size], s)
    fill = window_scores.min() if window_scores.size else 0.0
    points[~np.isfinite(points)] = fill
    return points


def best_point_adjusted_f1(
    scores: Sequence[float], truth: Sequence[bool], *, n_thresholds: int = 200
) -> dict:
    """
    Best point-adjusted F1 over a threshold sweep on point-level scores.

    Selects the threshold that maximizes point-adjusted F1. This is the standard
    SMAP / MSL "best F1" protocol used by telemanom and MEMTO, which makes those
    baselines comparable. Note that it selects the threshold using the labels, so
    it should be reported as an oracle-threshold upper bound, not a deployable
    operating point.

    Arguments:
        scores: Point-level anomaly scores (higher = more anomalous).
        truth: Point-level boolean ground truth.
        n_thresholds: Number of candidate thresholds sampled across the score range.
    Returns:
        dict: The best {'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'threshold'}.
    """
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_thresholds)))
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
    best["threshold"] = float("inf")
    for t in candidates:
        m = prf(point_adjust(scores >= t, truth), truth)
        if m["f1"] > best["f1"]:
            m["threshold"] = float(t)
            best = m
    return best
