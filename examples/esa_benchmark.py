"""
ESA-ADB Mission1 benchmark: the graph detectors on genuinely multivariate telemetry.

Separate from ``smap_benchmark.py`` rather than folded into it. SMAP and MSL are
one format and share every helper; ESA-ADB is a different format with real
timestamps, per-channel annotations, events split into disjoint segments, and a
published protocol of its own. Merging them would have meant rewriting the file
every SMAP number in the repository came from.

Why this dataset matters here: on SMAP a record is one sensor plus 24 near
constant command flags, so the graph detectors were only ever measured where
their graph is a lone self-loop or a relation over noise. Channels 41 to 46 are
one subsystem with a median cross-channel |r| of 0.86, and channel_44 runs
inverted against the other five. This is the first input the architecture was
actually designed for.

Protocol, fixed to ESA-ADB rather than chosen (arXiv:2406.17826, Methods):

* Splits come from :func:`~telemetry_anomdet.ingest.esa.esa_splits`: the mission
  is halved, the first half trains, and its last 3 months are validation.
* An event is an ``ID``, not a contiguous run. One event spans every channel it
  affects and may be several disjoint regions per channel, so that a burst of
  related disturbances counts once. Scoring regions separately would inflate
  false positives and deflate recall at the same time.
* Communication gaps are excluded; Anomaly and Rare Event are scored together.
* **F0.5 is the headline**, not F1: false alarms are what stop operators using
  these systems, so the benchmark weights precision. F1 is reported beside it.
* Metrics operate on binary detections. Threshold-agnostic scores such as
  PR-AUC are outside the comparable set and are not reported.

The subset is channels 41-46 because the benchmark's own full 58-channel results
defeat every algorithm it published, while the subset does not. Measured results
live in docs/source/user_guide/, not here.

Two things are measured that most published methods cannot provide, and that
ESA-ADB ranks as a *primary* aspect (2b, above timing and everything below it):

* **Channel identification.** ``channel_deviations`` names the channels that
  drove each alarm exactly, not by perturbation.
* **Alarm count per event.** Aspect 3, "exactly one detection per anomaly".

Usage::

    $env:TAD_ESA_DIR = ".../ESA-Mission1"
    uv run python examples/esa_benchmark.py

Configuration (env):
    TAD_ESA_DIR           -> extracted mission directory (required)
    TAD_ESA_CONFIGS       -> comma-separated config names (default: all)
    TAD_ESA_TRAIN_MONTHS  -> months at the END of the train split (default 12)
    TAD_ESA_TEST_MONTHS   -> months at the START of the test split (0 = all, default 12)
    TAD_ESA_WINDOW        -> window length in samples (default 30)
    TAD_ESA_STEP          -> stride between windows (default 3)
    TAD_ESA_RESAMPLE      -> pandas offset alias, empty for the native 30 s
    TAD_ESA_EMBED_DIM     -> detector width (default 4)
    TAD_ESA_GRID_SIZE     -> KAN spline intervals (default 3)
    TAD_ESA_TOPK          -> graph neighbours per node (default 5)
    TAD_ESA_LR            -> learning rate (default 0.00975)
    TAD_ESA_EPOCHS        -> training epochs (default 50)
    TAD_ESA_SMOOTHING     -> EWMA alpha over forecast errors (default 0.3)
    TAD_ESA_SEED          -> random seed (default 0)

``TAD_ESA_TRAIN_MONTHS`` and ``TAD_ESA_TEST_MONTHS`` default to slices rather
than the full splits: the full training half is 6.75 years at 30 s, about 7
million samples per channel. **A sliced test window is not comparable to the
published figures**, because it contains a different set of events. The evaluated
event count is printed for exactly that reason. Set both to 0 for the real thing.

Needs the deep extra (``uv sync --extra deep``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from telemetry_anomdet.feature_extraction.features import make_feature_table
from telemetry_anomdet.ingest import MISSION1_LIGHTWEIGHT, esa_splits, load_esa, load_esa_labels
from telemetry_anomdet.preprocessing import pipeline
from telemetry_anomdet.thresholding import detect_anomalies, threshold_for_budget

ROOT = Path(os.environ.get("TAD_ESA_DIR", "")).expanduser()
CHANNELS = list(MISSION1_LIGHTWEIGHT)

TRAIN_MONTHS = int(os.environ.get("TAD_ESA_TRAIN_MONTHS", "12"))
TEST_MONTHS = int(os.environ.get("TAD_ESA_TEST_MONTHS", "12"))
WINDOW = int(os.environ.get("TAD_ESA_WINDOW", "30"))
# Small by SMAP standards on purpose: the median annotated segment here is 13
# samples, so a stride of 10 can step straight over an event. This is a
# structural miss, not a model failure, and no hyperparameter recovers it.
STEP = int(os.environ.get("TAD_ESA_STEP", "3"))
# Training strides much further than scoring, and the two want different things.
# Detection needs a fine stride so no short event falls between windows; training
# needs varied examples, and neighbouring windows at stride 3 are 90% identical.
# At stride 3 a year of training is 350k windows, which is 270k optimiser steps
# per run for no added information.
TRAIN_STEP = int(os.environ.get("TAD_ESA_TRAIN_STEP", "30"))
RESAMPLE = os.environ.get("TAD_ESA_RESAMPLE", "").strip() or None
# The test split is 7 years at 30 s, 7.4 million samples per channel. Windowing
# it whole at stride 3 would allocate several gigabytes, so it is scored in
# chunks and the point scores stitched back together. Windows do not span a
# chunk boundary, so a chunk edge is blind for one window length: at 12 months a
# chunk that costs 7 blind windows out of a million.
CHUNK_MONTHS = int(os.environ.get("TAD_ESA_CHUNK_MONTHS", "12"))

EMBED_DIM = int(os.environ.get("TAD_ESA_EMBED_DIM", "4"))
GRID_SIZE = int(os.environ.get("TAD_ESA_GRID_SIZE", "3"))
TOPK = int(os.environ.get("TAD_ESA_TOPK", "5"))
LR = float(os.environ.get("TAD_ESA_LR", "0.00975"))
EPOCHS = int(os.environ.get("TAD_ESA_EPOCHS", "50"))
SMOOTHING = float(os.environ.get("TAD_ESA_SMOOTHING", "0.3")) or None
SEED = int(os.environ.get("TAD_ESA_SEED", "0"))

SELECTED = [n.strip() for n in os.environ.get("TAD_ESA_CONFIGS", "").split(",") if n.strip()]

# Operating points to evaluate each fitted model at. "dynamic" is
# dynamic_threshold, the setting used everywhere else in this repo; the numbers
# are alarm budgets passed to threshold_for_budget.
#
# The sweep exists because the first full run could not tell the architectures
# apart. Every model scored precision 1.000 with zero false positives, so F0.5
# was decided entirely by how many alarms the threshold let through: 39 spans
# gave 15 detections, 4 spans gave 3. That ranks threshold conservatism, not
# detection quality. Holding the alarm budget fixed puts every model at a
# comparable alarm rate, which is the only way a recall difference means
# anything. It is also the operationally honest control: operations can state
# how often a trigger may fire, and cannot state a recall they have no labels
# to measure.
BUDGETS = [
    b.strip()
    # Upper end informed by the dataset rather than guessed: 1.80% of Mission1's
    # points are annotated, so a budget much past 0.02 flags more than the
    # anomalies present and precision has to fall. In the first sweep precision
    # was still 1.000 at a budget of 0.005, which is where the headroom is.
    for b in os.environ.get("TAD_ESA_BUDGETS", "dynamic,0.001,0.005,0.01,0.02").split(",")
    if b.strip()
]

# Whether to apply telemanom's sequence pruning, which drops a candidate whose
# peak error is not min_decrease above the next one down.
#
# Swept rather than fixed because on this dataset it, not the threshold,
# is what sets the alarm count. Budgets from 0.0002 to 0.002 all collapsed to a
# single surviving span: raising the budget admits hundreds of candidates and
# pruning then discards all but the largest. That is reasonable on SMAP, where a
# channel has a handful of long anomalies, and wrong here, where 65 events are
# scattered over 7 years and 62% of segments are shorter than 30 samples. Rows
# marked "np" skip it.
PRUNE_MODES = [
    p.strip() == "1"
    for p in os.environ.get("TAD_ESA_PRUNE", "1,0").split(",")
    if p.strip() in ("0", "1")
]


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def make_random():
    """Uniform random scores: the floor every other row has to clear."""
    from telemetry_anomdet.models.base import BaseDetector

    class RandomScorer(BaseDetector):
        def fit(self, X, y=None):
            rng = np.random.default_rng(SEED)
            self._set_post_fit(rng.random(len(X)))
            return self

        def decision_function(self, X):
            return np.random.default_rng(SEED + 1).random(len(X))

    return RandomScorer(percentile=95.0)


def make_gdn():
    """Plain GDN. The only architecture with a working C emitter today."""
    from telemetry_anomdet.models.deep import GDN

    return GDN(
        embed_dim=EMBED_DIM,
        topk=TOPK,
        lr=LR,
        epochs=EPOCHS,
        random_state=SEED,
        smoothing=SMOOTHING,
    )


def make_kangdn(feat_mode: str):
    from telemetry_anomdet.models.deep import KANGDN

    return KANGDN(
        embed_dim=EMBED_DIM,
        topk=TOPK,
        lr=LR,
        epochs=EPOCHS,
        random_state=SEED,
        grid_size=GRID_SIZE,
        smoothing=SMOOTHING,
        feat_mode=feat_mode,
    )


# Every feature transform the toolkit offers, plus plain GDN and the random
# floor. `kan_residual` was benchmarked here too and then removed: it was never
# best in any of the six regimes measured across SMAP, MSL and this dataset, and
# on ESA it only tied the plain linear transform it exists to improve on.
#
# `ar_kan` was on notice for the opposite reason and survived. It was worst or
# second worst in all four SMAP and MSL measurements, but neither dataset gives
# a graph anything to learn, and on six correlated sensors it is the best model
# here. Benchmarking before deleting is why it is still in the table.
CONFIGS: dict[str, object] = {
    "random": make_random,
    "gdn": make_gdn,
    "kangdn": lambda: make_kangdn("linear"),
    "kanfeat": lambda: make_kangdn("kan"),
    "arkan": lambda: make_kangdn("ar_kan"),
    "arkanres": lambda: make_kangdn("ar_kan_residual"),
}


# ---------------------------------------------------------------------------
# Windows and scores
# ---------------------------------------------------------------------------


def window_period(lo: pd.Timestamp, hi: pd.Timestamp, step: int):
    """
    Load one period and window it.

    Returns:
        tuple: ``(X, timestamps)`` where ``X`` is
            ``(n_windows, window_size, n_channels)`` with channel order fixed to
            :data:`CHANNELS`, and ``timestamps`` indexes the underlying samples.
    """
    long = load_esa(ROOT, CHANNELS, start=lo, end=hi, resample_rule=RESAMPLE).to_pandas()
    long = pipeline(long, resample_rule=None)
    timestamps = np.sort(long["timestamp"].unique())
    # variables= pins column order, so feature index i is CHANNELS[i] and the
    # per-channel attribution below can name a real channel.
    X = make_feature_table(long, variables=CHANNELS, window_size=WINDOW, step=step)
    return X, pd.DatetimeIndex(timestamps)


def train_period(months: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The tail of the training split, so the fit sees the most recent behaviour."""
    lo, hi = esa_splits()["train"]
    return (max(lo, hi - pd.DateOffset(months=months)) if months else lo), hi


def test_chunks(months: int):
    """Yield ``(lo, hi)`` bounds covering the test split in bounded pieces."""
    lo, hi = esa_splits()["test"]
    if months:
        hi = min(hi, lo + pd.DateOffset(months=months))
    cursor = lo
    while cursor < hi:
        nxt = min(hi, cursor + pd.DateOffset(months=CHUNK_MONTHS))
        yield cursor, nxt
        cursor = nxt


def window_scores_to_points(scores: np.ndarray, blame: np.ndarray, n_points: int):
    """
    Spread per-window scores back over the samples each window covers.

    A sample takes the maximum score of any window containing it, and inherits
    that window's most deviating channel. Max rather than mean because a short
    event occupies a minority of its window's samples, and averaging would
    dilute precisely the events this dataset is made of.

    Returns:
        tuple: ``(points, point_blame)``, both length ``n_points``.
            ``point_blame`` is the channel index behind each sample's score, or
            -1 where no window reached it. Carrying blame per sample rather than
            keeping the full ``(n_windows, n_channels)`` deviation matrix is what
            makes chunked scoring of a 7-year split affordable.
    """
    points = np.full(n_points, -np.inf, dtype=float)
    point_blame = np.full(n_points, -1, dtype=np.int8)

    for i, score in enumerate(scores):
        lo = i * STEP
        hi = min(lo + WINDOW, n_points)
        if hi <= lo:
            break
        better = score > points[lo:hi]
        points[lo:hi] = np.where(better, score, points[lo:hi])
        point_blame[lo:hi] = np.where(better, blame[i], point_blame[lo:hi])

    # Trailing samples no window covered (the series rarely divides evenly).
    uncovered = ~np.isfinite(points)
    points[uncovered] = np.min(points[~uncovered]) if (~uncovered).any() else 0.0
    return points, point_blame


def predicted_intervals(
    point_scores: np.ndarray,
    timestamps: pd.DatetimeIndex,
    setting: str,
    prune: bool,
):
    """
    Label-free detection at one operating point, returned as timestamp intervals.

    Arguments:
        point_scores: One score per sample.
        timestamps: Sample timestamps, same length.
        setting: ``"dynamic"`` for :func:`dynamic_threshold`, otherwise a float
            alarm budget for :func:`threshold_for_budget`.
        prune: Apply telemanom's sequence pruning.

    Notes:
        Both routes choose the cutoff from the score distribution alone. No
        variant here looks at a label, because an oracle threshold is not one
        anyone can deploy.
    """
    if setting == "dynamic":
        found = detect_anomalies(point_scores, prune=prune)
    else:
        chosen = threshold_for_budget(point_scores, budget=float(setting))
        found = detect_anomalies(point_scores, threshold=chosen["threshold"], prune=prune)

    spans = [
        (timestamps[start], timestamps[min(end, len(timestamps) - 1)])
        for start, end in found["sequences"]
    ]
    return spans, found


# ---------------------------------------------------------------------------
# Event-level scoring, ESA-ADB style
# ---------------------------------------------------------------------------


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    """Closed-interval overlap."""
    return a_start <= b_end and b_start <= a_end


def score_events(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    labels: pd.DataFrame,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> dict:
    """
    Event-wise precision, recall, F0.5 and F1, grouped by event ID.

    Arguments:
        spans: Predicted anomalous intervals.
        labels: Segment rows from
            :func:`~telemetry_anomdet.ingest.esa.load_esa_labels`.
        window: The evaluated period; segments are clipped to it and events
            falling entirely outside it are not counted against recall.
    Returns:
        dict: counts and scores, plus ``alarms_per_event`` for ESA-ADB's aspect
        3 ("exactly one detection per anomaly").

    Notes:
        An event counts as detected when any predicted interval overlaps **any**
        of its segments, on any affected channel. A predicted interval that
        overlaps no segment of any event is one false positive. This is the
        grouping the dataset was annotated for; treating each segment as its own
        event would both invent false positives and invent misses.
    """
    lo, hi = window
    in_window = labels[(labels["EndTime"] >= lo) & (labels["StartTime"] < hi)]
    events = {
        event_id: list(zip(group["StartTime"], group["EndTime"], strict=True))
        for event_id, group in in_window.groupby("ID")
    }

    detected, alarms_per_event = set(), []
    for event_id, segments in events.items():
        hits = sum(
            any(_overlaps(s, e, seg_start, seg_end) for seg_start, seg_end in segments)
            for s, e in spans
        )
        if hits:
            detected.add(event_id)
            alarms_per_event.append(hits)

    matched_spans = sum(
        any(
            _overlaps(s, e, seg_start, seg_end)
            for segments in events.values()
            for seg_start, seg_end in segments
        )
        for s, e in spans
    )

    tp, fn = len(detected), len(events) - len(detected)
    fp = len(spans) - matched_spans
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    def f_beta(beta: float) -> float:
        b2 = beta * beta
        denom = b2 * precision + recall
        return (1 + b2) * precision * recall / denom if denom else 0.0

    return {
        "events": len(events),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f05": f_beta(0.5),
        "f1": f_beta(1.0),
        "alarms_per_event": float(np.mean(alarms_per_event)) if alarms_per_event else 0.0,
    }


def score_channel_identification(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]],
    point_blame: np.ndarray,
    timestamps: pd.DatetimeIndex,
    labels: pd.DataFrame,
) -> dict:
    """
    Channel identification: which channels did the detector blame, and was it right?

    ESA-ADB ranks this a *primary* aspect, above detection timing, because an
    algorithm that cannot name affected channels is of little practical use to
    an operator and deepens the black-box problem. ``channel_deviations`` answers
    it exactly rather than by perturbation, which is unusual: the unsupervised
    baselines in the paper cannot report it at all and their table leaves those
    cells blank.

    Only spans that overlap an annotation are assessed. A span overlapping
    nothing is already a false positive in :func:`score_events`, and naming a
    channel for a non-event is not a channel identification error.

    Returns:
        dict: Micro-averaged precision and recall over (span, channel) pairs.
    """
    hits = misses = spurious = 0

    for span_start, span_end in spans:
        overlapping = labels[(labels["EndTime"] >= span_start) & (labels["StartTime"] <= span_end)]
        if overlapping.empty:
            continue

        truth = set(overlapping["Channel"])
        lo = int(np.searchsorted(timestamps, span_start))
        hi = int(np.searchsorted(timestamps, span_end, side="right"))
        blamed = {CHANNELS[i] for i in np.unique(point_blame[lo:hi]) if i >= 0}

        hits += len(blamed & truth)
        spurious += len(blamed - truth)
        misses += len(truth - blamed)

    predicted, actual = hits + spurious, hits + misses
    return {
        "channel_precision": hits / predicted if predicted else 0.0,
        "channel_recall": hits / actual if actual else 0.0,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def score_test_split(detector) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Score the whole test split chunk by chunk.

    The chunks are contiguous and their point scores are concatenated, so the
    threshold is still chosen once over the entire test period rather than per
    chunk. A per-chunk threshold would be a different, easier problem.
    """
    all_points, all_blame, all_ts = [], [], []
    for lo, hi in test_chunks(TEST_MONTHS):
        X, timestamps = window_period(lo, hi, STEP)
        if X.size == 0:
            continue

        if hasattr(detector, "channel_deviations"):
            # One forward pass, not two. decision_function is exactly
            # _deviation_score(_errors_for(X)), which is the row-wise maximum of
            # these same normalised deviations over the scoring channels, so
            # calling both would forward every window twice for no new
            # information. That doubling was roughly half the runtime of the
            # first full pass over this split.
            deviations = detector.channel_deviations(X)
            scoring = detector.scoring_channels_
            scores = deviations[:, scoring].max(axis=1)
            blame = deviations.argmax(axis=1).astype(np.int8)
        else:
            scores = detector.decision_function(X)
            blame = np.full(len(scores), -1, dtype=np.int8)

        points, point_blame = window_scores_to_points(scores, blame, len(timestamps))
        all_points.append(points)
        all_blame.append(point_blame)
        all_ts.append(timestamps)

    return (
        np.concatenate(all_points),
        np.concatenate(all_blame),
        pd.DatetimeIndex(np.concatenate(all_ts)),
    )


def run_config(name: str, factory, X_train: np.ndarray, labels: pd.DataFrame) -> list[dict]:
    """
    Fit once, score the test split once, then evaluate at every operating point.

    Scoring 7 years at stride 3 is by far the expensive part, and it does not
    depend on the threshold. Sweeping budgets over the cached point scores costs
    almost nothing, so the sweep is close to free once a model is trained.
    """
    detector = factory()
    detector.fit(X_train)
    points, blame, test_ts = score_test_split(detector)

    results = []
    for setting in BUDGETS:
        for prune in PRUNE_MODES:
            spans, found = predicted_intervals(points, test_ts, setting, prune)
            result = score_events(spans, labels, (test_ts[0], test_ts[-1]))
            result["name"] = name
            result["setting"] = setting if prune else f"{setting} np"
            result["spans"] = len(spans)
            result["threshold"] = found["threshold"]
            result.update(score_channel_identification(spans, blame, test_ts, labels))
            results.append(result)
    return results


def _row(r: dict) -> str:
    return (
        f"{r['name']:>10s} {r['setting']:>9s} {r['f05']:7.3f} {r['f1']:7.3f} "
        f"{r['precision']:7.3f} {r['recall']:7.3f} {r['tp']:4d} {r['fp']:5d} "
        f"{r['fn']:4d} {r['channel_recall']:6.3f} {r['alarms_per_event']:6.1f} "
        f"{r['spans']:6d}"
    )


def main() -> None:
    if not ROOT or not (ROOT / "channels").is_dir():
        raise SystemExit("Set TAD_ESA_DIR to an extracted ESA mission directory.")

    names = SELECTED or list(CONFIGS)
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown config(s): {', '.join(unknown)}. Known: {', '.join(CONFIGS)}")

    labels = load_esa_labels(ROOT, CHANNELS)

    print(f"ESA-ADB Mission1, channels {CHANNELS[0]}..{CHANNELS[-1]}")
    print(f"window={WINDOW} step={STEP} (train {TRAIN_STEP}) resample={RESAMPLE or 'native 30s'}")

    train_lo, train_hi = train_period(TRAIN_MONTHS)
    X_train, _ = window_period(train_lo, train_hi, TRAIN_STEP)
    print(f"train        : {train_lo.date()} to {train_hi.date()}   windows: {len(X_train):,}")

    test_lo = esa_splits()["test"][0]
    test_hi = list(test_chunks(TEST_MONTHS))[-1][1]
    scored = labels[(labels["EndTime"] >= test_lo) & (labels["StartTime"] < test_hi)]
    print(f"test         : {test_lo.date()} to {test_hi.date()}   events: {scored.ID.nunique()}")
    if TEST_MONTHS:
        print("  (sliced test window: NOT comparable to the published figures)")

    header = (
        f"{'config':>10s} {'setting':>9s} {'F0.5':>7s} {'F1':>7s} {'P':>7s} {'R':>7s} "
        f"{'TP':>4s} {'FP':>5s} {'FN':>4s} {'chR':>6s} {'al/ev':>6s} {'spans':>6s}"
    )
    results = []
    for name in names:
        print(f"\n=== {name} ===")
        print(header)
        for r in run_config(name, CONFIGS[name], X_train, labels):
            results.append(r)
            print(_row(r))

    print("\n" + "=" * 96)
    print("BEST OPERATING POINT PER CONFIG (by F0.5)")
    print(header)
    print("-" * 96)
    best = {}
    for r in results:
        if r["name"] not in best or r["f05"] > best[r["name"]]["f05"]:
            best[r["name"]] = r
    for r in sorted(best.values(), key=lambda r: -r["f05"]):
        print(_row(r))

    print(
        "\nF0.5 is the headline: ESA-ADB weights precision because false alarms\n"
        "are the main obstacle to operational adoption. Reference on these six\n"
        "channels, full test split: Telemanom-ESA-Pruned F0.5 0.786 (P 0.999 R 0.424).\n"
        "\n"
        "Read the sweep down a column, not across configs at one setting. Compare\n"
        "models at a MATCHED budget: at a fixed alarm rate a recall difference is\n"
        "a detection difference. The dynamic row is kept for continuity with the\n"
        "SMAP and MSL numbers, but it gives each model a different alarm rate and\n"
        "so cannot separate detection quality from threshold conservatism.\n"
        "\n"
        "chR is channel identification, a primary aspect of the benchmark that the\n"
        "paper's unsupervised baselines cannot report at all. chP is omitted from\n"
        "this table: most events here affect all six channels, so almost any\n"
        "channel named is correct and it sat at 1.000 for every model."
    )


if __name__ == "__main__":
    main()
