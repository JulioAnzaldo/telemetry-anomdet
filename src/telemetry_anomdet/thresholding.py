# src/telemetry_anomdet/thresholding.py

"""
Label-free thresholding of an anomaly error signal.

Every threshold this toolkit produced previously came from one of two places: a
percentile of the training scores, or a sweep chosen against the test labels.
The first ignores the shape of the error distribution; the second is an oracle
and cannot be deployed. This module implements the nonparametric dynamic
thresholding and pruning of Hundman et al. (2018), which select an operating
point from the error signal alone.

The method has two stages.

**Threshold selection.** Candidate thresholds are drawn from
``eps = mu + z * sigma`` over a range of ``z``. The chosen candidate maximises

.. math::

    \\frac{\\Delta\\mu/\\mu + \\Delta\\sigma/\\sigma}{|e_a| + |E_{seq}|^2}

where ``e_a`` are the errors above the candidate, ``E_seq`` the contiguous runs
they form, and ``delta_mu``, ``delta_sigma`` the reductions in the mean and
standard deviation of the error signal once those errors are removed.

The numerator asks how much calmer the signal becomes when the flagged points
are taken out: a large reduction means they really were outliers rather than
part of the bulk. The denominator prices that reduction. ``|e_a|`` penalises
flagging many points, and ``|E_seq|**2`` penalises fragmentation quadratically,
so a few coherent events are strongly preferred over the same number of points
scattered across the series. The ratio is therefore a signal-to-cost trade
resolved without reference to any label.

**Pruning.** Candidate sequences are ranked by their peak error and a sequence
is kept only while consecutive peaks fall off gradually. A large relative drop
marks the boundary between genuine anomalies and the nominal tail, and
everything below the last such drop is reclassified as nominal. The nominal
maximum is appended to the ranking so that a single candidate still has
something to be compared against.

All functions here operate on a plain error array, so the caller decides which
errors to pass: training errors for a fixed operating point to deploy, or a
trailing window for an adaptive one.

Choosing between the two selectors: :func:`threshold_for_budget` is exactly
controllable and suits a deployed trigger, where operations can state an alarm
rate but cannot state a recall they have no way to observe.
:func:`dynamic_threshold` needs no budget chosen in advance.

Two defaults depart from the published method, both because the error signals
differ: the candidate range starts lower (see :func:`dynamic_threshold`), and
pruning is off. One interaction is worth knowing before deploying a budget: at
small budgets the flagged points are scattered singletons that never form a run
of two, so :func:`filter_sequences` with its default minimum length discards all
of them.

Measured results on SMAP, and how this module compares with the published
implementation, are recorded in the anomaly scoring page of the documentation
rather than here, so that they can be revised as the detectors change.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "anomalous_sequences",
    "dynamic_threshold",
    "threshold_for_budget",
    "filter_sequences",
    "prune_sequences",
    "detect_anomalies",
]


def threshold_for_budget(errors: Sequence[float], budget: float = 0.01) -> dict:
    """
    Threshold that flags approximately ``budget`` of the signal.

    The alarm budget is the operationally meaningful control. Operations can say
    how often a trigger may fire; they cannot say what recall they want, because
    recall is unobservable without labels. The budget needs no search: it is a
    quantile of the errors.

    When anomalies are a small part of the signal, the flagged fraction tracks
    the false alarm rate closely, so choosing a budget sets the false alarm rate
    directly and without labels.

    The fraction achieved is approximate rather than bounded, and can exceed the
    budget slightly. Read ``flagged`` in the result for the fraction actually
    reached rather than assuming the budget was met exactly.

    Arguments:
        errors: Error signal, one value per timestep.
        budget: Target fraction of points to flag, in (0, 1).
    Returns:
        dict: ``threshold``, the ``flagged`` fraction actually achieved, and the
        ``n_above`` and ``n_sequences`` it produces.
    """
    if not 0.0 < budget < 1.0:
        raise ValueError(f"budget must lie in (0, 1), got {budget}")
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        raise ValueError("errors must not be empty")

    # TODO(0.3.0): switch to method="higher" and restore the "no more than the
    # budget" guarantee. numpy's default linear interpolation puts the cutoff
    # between two samples, so slightly more than the budget can sit above it:
    # 50 points at budget=0.05 flags 3 (0.06) where 2 (0.04) would fit.
    # method="higher" snaps to a real sample and holds the bound. Deferred
    # because it shifts the flagged set by one point and the benchmark reads
    # this, so it lands with the 0.3.0 rerun (SMAP, MSL, ESA-ADB) rather than
    # moving the published numbers twice.
    threshold = float(np.quantile(errors, 1.0 - budget))
    above = errors > threshold
    return {
        "threshold": threshold,
        "flagged": float(above.mean()),
        "n_above": int(above.sum()),
        "n_sequences": len(anomalous_sequences(errors, threshold)),
    }


def anomalous_sequences(errors: Sequence[float], threshold: float) -> list[tuple[int, int]]:
    """
    Contiguous runs of errors strictly above a threshold.

    Arguments:
        errors: Error signal, one value per timestep.
        threshold: Cutoff; a value must exceed it to be included.
    Returns:
        list: ``(start, end)`` index pairs, inclusive of both ends.
    """
    flags = np.asarray(errors, dtype=float) > threshold
    if not flags.any():
        return []

    # Run boundaries are where the flag changes, padded so runs touching either
    # end of the series are closed properly.
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b - 1)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _candidate_thresholds(
    errors: np.ndarray,
    strategy: str,
    n_candidates: int,
    z_range: tuple[float, float],
    q_range: tuple[float, float],
) -> np.ndarray:
    """
    Threshold candidates to evaluate, by strategy. See :func:`dynamic_threshold`.
    """
    if strategy == "quantile":
        qs = np.linspace(q_range[0], q_range[1], n_candidates)
        return np.unique(np.quantile(errors, qs))
    if strategy == "sigma":
        zs = np.linspace(z_range[0], z_range[1], n_candidates)
        return float(errors.mean()) + zs * float(errors.std())
    raise ValueError(f"strategy must be 'quantile' or 'sigma', got {strategy!r}")


def dynamic_threshold(
    errors: Sequence[float],
    strategy: str = "sigma",
    n_candidates: int = 120,
    z_range: tuple[float, float] = (1.0, 12.0),
    q_range: tuple[float, float] = (0.80, 0.9999),
) -> dict:
    """
    Choose a threshold from the error signal alone, with no labels.

    Each candidate is scored by the objective described in the module docstring:
    the reduction it produces in the mean and standard deviation of the error
    signal, divided by the number of points and the square of the number of
    sequences it flags.

    Two candidate sets are available.

    ``"sigma"`` (default)
        ``mu + z * sigma`` for ``z`` spanning ``z_range``, as published.

    ``"quantile"``
        Evenly spaced upper quantiles of the errors, spanning ``q_range``.

    Which set is better depends on the signal, and the measured difference is
    small, so the published choice is the default.

    The lower end of ``z_range`` matters more, and the default of 1.0 departs
    from the published 2.5 deliberately. Deviation scores here are already
    normalised by each node's training median and IQR, so their distribution is
    much tighter than the smoothed prediction errors the original method was
    tuned on, and a floor of 2.5 sits past the region where useful thresholds
    lie. The choice was validated on held-out channels; see the anomaly scoring
    page of the documentation.

    Arguments:
        errors: Error signal, one value per timestep.
        strategy: ``"quantile"`` or ``"sigma"``.
        n_candidates: How many candidates to evaluate.
        z_range: Range of ``z`` for the sigma strategy.
        q_range: Range of quantiles for the quantile strategy.
    Returns:
        dict: ``threshold``, the objective ``score``, the ``n_above`` and
        ``n_sequences`` it flags, and the ``strategy`` used. When no candidate
        flags anything, ``threshold`` is the maximum error so that nothing
        exceeds it and ``score`` is 0.0.
    """
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        raise ValueError("errors must not be empty")

    mu = float(errors.mean())
    sigma = float(errors.std())
    none_found = {
        "threshold": float(errors.max()),
        "score": 0.0,
        "n_above": 0,
        "n_sequences": 0,
        "strategy": strategy,
    }
    # A flat signal has no outliers to find, and the objective would divide by
    # zero on both terms.
    if sigma == 0.0 or mu == 0.0:
        _candidate_thresholds(errors, strategy, 1, z_range, q_range)  # validate strategy
        return none_found

    best = none_found
    for candidate in _candidate_thresholds(errors, strategy, n_candidates, z_range, q_range):
        above = errors > candidate
        n_above = int(above.sum())
        if n_above == 0 or n_above == errors.size:
            continue

        below = errors[~above]
        delta_mu = mu - float(below.mean())
        delta_sigma = sigma - float(below.std())
        n_sequences = len(anomalous_sequences(errors, candidate))

        score = (delta_mu / mu + delta_sigma / sigma) / (n_above + n_sequences**2)
        if score > best["score"]:
            best = {
                "threshold": float(candidate),
                "score": float(score),
                "n_above": n_above,
                "n_sequences": n_sequences,
                "strategy": strategy,
            }
    return best


def filter_sequences(
    sequences: list[tuple[int, int]],
    min_length: int = 2,
    ignore_before: int = 0,
) -> list[tuple[int, int]]:
    """
    Drop predictions that the detection protocol treats as unusable.

    Two filters, both taken from telemanom, applied to sequences before they are
    scored or acted on.

    ``min_length`` discards runs shorter than the given length. A single
    isolated sample above the threshold is a spike in the error signal rather
    than an event, and telemanom never promotes one to a sequence.

    ``ignore_before`` discards sequences ending before the given index. A
    forecaster has no history at the start of a stream, so its errors there
    reflect the cold start rather than the data. telemanom skips the opening
    ``2 * l_s`` samples, halving that for shorter streams and skipping nothing
    for very short ones.

    Arguments:
        sequences: Predicted ``(start, end)`` ranges, inclusive.
        min_length: Shortest run to keep, in samples.
        ignore_before: Index before which sequences are discarded.
    Returns:
        list: The retained subset, in the original order.
    """
    return [(a, b) for a, b in sequences if (b - a + 1) >= min_length and b >= ignore_before]


def startup_skip(n_samples: int, context: int) -> int:
    """
    How many opening samples to ignore, following telemanom's rule.

    Errors at the start of a stream reflect the model's lack of history. The
    published rule skips twice the context length, halves that when the stream
    is under 2500 samples, and skips nothing under 1800, so that short streams
    are not discarded entirely.

    Arguments:
        n_samples: Length of the scored stream.
        context: Model context length in samples.
    Returns:
        int: Index before which predictions should be ignored.
    """
    if n_samples < 1800:
        return 0
    return context if n_samples < 2500 else context * 2


def prune_sequences(
    errors: Sequence[float],
    sequences: list[tuple[int, int]],
    threshold: float,
    min_decrease: float = 0.13,
) -> list[tuple[int, int]]:
    """
    Drop candidate sequences that are not clearly separated from the nominal tail.

    Sequences are ranked by peak error, the nominal maximum is appended, and the
    relative drop between consecutive peaks is examined. Everything below the
    last drop exceeding ``min_decrease`` is reclassified as nominal: a gradual
    decline means the remaining candidates are part of the same population,
    whereas a sharp fall marks a real boundary.

    Arguments:
        errors: Error signal the sequences were found in.
        sequences: Candidate ``(start, end)`` pairs from :func:`anomalous_sequences`.
        threshold: The threshold used to find them, needed to identify the
            nominal population.
        min_decrease: Relative drop that counts as a boundary, as a fraction.
    Returns:
        list: The retained subset of ``sequences``, in their original order.
    """
    if not sequences:
        return []
    errors = np.asarray(errors, dtype=float)

    peaks = np.array([errors[start : end + 1].max() for start, end in sequences])
    order = np.argsort(-peaks, kind="stable")

    below = errors[errors <= threshold]
    nominal_max = float(below.max()) if below.size else 0.0
    ranked = np.concatenate((peaks[order], [nominal_max]))

    # Walk the ranking and remember the last position where the peak falls away
    # sharply. Everything above that position is kept.
    keep_count = 0
    for i in range(ranked.size - 1):
        if ranked[i] <= 0.0:
            continue
        drop = (ranked[i] - ranked[i + 1]) / ranked[i]
        if drop > min_decrease:
            keep_count = i + 1

    kept = sorted(int(idx) for idx in order[:keep_count])
    return [sequences[i] for i in kept]


def detect_anomalies(
    errors: Sequence[float],
    threshold: float | None = None,
    min_decrease: float = 0.13,
    prune: bool = True,
    **threshold_kwargs,
) -> dict:
    """
    Full label-free detection: choose a threshold, find sequences, prune them.

    Arguments:
        errors: Error signal, one value per timestep.
        threshold: Use this cutoff instead of selecting one. Useful for applying
            an operating point derived from training errors to new data.
        min_decrease: Passed to :func:`prune_sequences`.
        prune: Set False to keep every sequence above the threshold.
        **threshold_kwargs: Passed to :func:`dynamic_threshold`.
    Returns:
        dict: ``mask`` (boolean, one entry per timestep), ``threshold``,
        ``sequences`` retained, and ``n_pruned`` sequences discarded.
    """
    errors = np.asarray(errors, dtype=float)
    if threshold is None:
        chosen = dynamic_threshold(errors, **threshold_kwargs)
        threshold = chosen["threshold"]

    found = anomalous_sequences(errors, threshold)
    kept = prune_sequences(errors, found, threshold, min_decrease) if prune else found

    mask = np.zeros(errors.shape, dtype=bool)
    for start, end in kept:
        mask[start : end + 1] = True

    return {
        "mask": mask,
        "threshold": float(threshold),
        "sequences": kept,
        "n_pruned": len(found) - len(kept),
    }
