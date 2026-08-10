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


# ---------------------------------------------------------------------------
# Threshold-free and operating-point metrics
# ---------------------------------------------------------------------------


def pr_auc(scores: Sequence[float], truth: Sequence[bool]) -> float:
    """
    Area under the precision-recall curve, as average precision.

    Computed on raw point scores with no point adjustment, so a detector is
    credited for the points it actually flags. Two properties make this the
    honest companion to :func:`best_point_adjusted_f1`:

    * It integrates over every operating point instead of reporting the single
      best one, so no threshold can be selected against the labels.
    * Its value for an uninformative detector is the positive base rate. Any
      score above that reflects real ranking ability, and the margin is
      interpretable. ROC AUC instead sits at 0.5 for random regardless of class
      balance, which flatters a detector when anomalies are rare.

    Arguments:
        scores: Point-level anomaly scores (higher = more anomalous).
        truth: Point-level boolean ground truth, same length as ``scores``.
    Returns:
        float: Average precision in [0, 1]; the base rate for random scores.
    """
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    if scores.shape != truth.shape:
        raise ValueError(f"scores and truth must match: {scores.shape} vs {truth.shape}")
    n_pos = int(truth.sum())
    if n_pos == 0 or n_pos == truth.size:
        return float(n_pos) / float(truth.size) if truth.size else 0.0

    order = np.argsort(-scores, kind="stable")
    hits = truth[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, hits.size + 1)
    # Average precision: the mean precision at each rank holding a true positive,
    # which equals the sum of precision * (change in recall).
    return float(precision[hits].sum() / n_pos)


def false_alarm_rate_at_recall(
    scores: Sequence[float], truth: Sequence[bool], target_recall: float = 0.8
) -> dict:
    """
    Cost of reaching a recall target, as a false positive rate per point.

    Answers the operational question a fixed threshold has to settle: to catch
    this fraction of anomalous points, how often does the detector fire on
    nominal data? Unlike a best-F1 figure this is reported at a stated recall,
    so two detectors are compared at the same sensitivity.

    Arguments:
        scores: Point-level anomaly scores (higher = more anomalous).
        truth: Point-level boolean ground truth, same length as ``scores``.
        target_recall: Recall to reach, in (0, 1].
    Returns:
        dict: ``{'threshold', 'recall', 'false_alarm_rate', 'precision'}``. The
        false alarm rate is false positives divided by the number of nominal
        points. Returns a rate of 1.0 when the target recall is unreachable.
    """
    if not (0.0 < target_recall <= 1.0):
        raise ValueError(f"target_recall must lie in (0, 1], got {target_recall}")
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    if scores.shape != truth.shape:
        raise ValueError(f"scores and truth must match: {scores.shape} vs {truth.shape}")

    n_pos = int(truth.sum())
    n_neg = int(truth.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {"threshold": float("inf"), "recall": 0.0, "false_alarm_rate": 1.0, "precision": 0.0}

    order = np.argsort(-scores, kind="stable")
    hits = truth[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(~hits)
    recall = tp / n_pos

    reached = np.flatnonzero(recall >= target_recall)
    if reached.size == 0:
        return {
            "threshold": float(scores.min()),
            "recall": float(recall[-1]),
            "false_alarm_rate": 1.0,
            "precision": float(n_pos) / truth.size,
        }
    k = int(reached[0])
    denom = tp[k] + fp[k]
    return {
        "threshold": float(scores[order][k]),
        "recall": float(recall[k]),
        "false_alarm_rate": float(fp[k]) / float(n_neg),
        "precision": float(tp[k]) / float(denom) if denom else 0.0,
    }


# ---------------------------------------------------------------------------
# Event-level scoring, matching the telemanom protocol
# ---------------------------------------------------------------------------


def evaluate_sequences(
    predicted: Sequence[tuple[int, int]],
    true_sequences: Sequence[tuple[int, int]],
) -> dict:
    """
    Score predicted anomaly ranges against labelled ones, event by event.

    This reproduces the scoring in Hundman et al.'s telemanom, so numbers are
    directly comparable with the results published for SMAP and MSL. It is the
    metric an operator experiences: how many real events were caught, and how
    many times the system cried wolf.

    The two sides are counted over different things, which is deliberate and
    easy to get wrong. A **true positive** is a labelled sequence that some
    prediction overlapped, so true positives and false negatives partition the
    labelled sequences. A **false positive** is a *predicted* sequence that
    overlapped nothing, so it is counted over predictions instead.

    When one prediction spans several labelled sequences, only the first is
    credited. A single alarm covering two events is one catch, not two.

    Contrast :func:`point_adjusted_f1`, which counts points and credits an
    entire labelled segment to a single flagged sample. That inflates a detector
    that raises many short false alarms beside a few long true ones: the same
    detector can read 0.71 point-adjusted and 0.13 here.

    Arguments:
        predicted: Predicted ``(start, end)`` ranges, inclusive of both ends.
        true_sequences: Labelled ``(start, end)`` ranges, inclusive.
    Returns:
        dict: ``true_positives``, ``false_positives``, ``false_negatives``,
        and the ``tp_sequences`` / ``fp_sequences`` that produced them.
    """

    def overlaps(a, b):
        return not (a[1] < b[0] or a[0] > b[1])

    matched: list[int] = []
    tp_sequences: list[tuple[int, int]] = []
    fp_sequences: list[tuple[int, int]] = []

    for seq in predicted:
        hits = [i for i, true in enumerate(true_sequences) if overlaps(seq, true)]
        if hits:
            tp_sequences.append(tuple(seq))
            # Only the first labelled sequence a prediction reaches is credited.
            if hits[0] not in matched:
                matched.append(hits[0])
        else:
            fp_sequences.append(tuple(seq))

    return {
        "true_positives": len(matched),
        "false_positives": len(fp_sequences),
        "false_negatives": len(true_sequences) - len(matched),
        "tp_sequences": tp_sequences,
        "fp_sequences": fp_sequences,
    }


def sequence_prf(rows: Sequence[dict]) -> dict:
    """
    Aggregate per-channel :func:`evaluate_sequences` results, as telemanom does.

    Counts are pooled across channels before precision and recall are computed,
    rather than averaging per-channel rates. Channels that detect nothing then
    contribute their misses without also contributing a precision of zero.

    Arguments:
        rows: Per-channel dicts from :func:`evaluate_sequences`.
    Returns:
        dict: pooled ``true_positives``, ``false_positives``,
        ``false_negatives``, and the derived ``precision``, ``recall``, ``f1``.
    """
    tp = sum(int(r["true_positives"]) for r in rows)
    fp = sum(int(r["false_positives"]) for r in rows)
    fn = sum(int(r["false_negatives"]) for r in rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
