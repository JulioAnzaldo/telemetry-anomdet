"""
SMAP classical-baseline benchmark.

Runs the PCA + KMeans ensemble across SMAP channels and reports point-adjusted
F1, the standard SMAP metric (comparable to telemanom and MEMTO). For each
channel it trains on the nominal train split and scores the test split.

Scores are swept over a range of thresholds and the best point-adjusted F1 is
reported. That is the standard SMAP "best F1" protocol (telemanom / MEMTO report
their numbers the same way), which makes the comparison fair. Note the threshold
is selected against the labels, so treat it as an oracle-threshold upper bound,
not a deployable operating point.

Point it at a local copy of the SMAP dataset (telemanom format), same as
examples/smap_demo.py:
    TAD_SMAP_DIR    -> directory containing train/ and test/ .npy files
    TAD_SMAP_LABELS -> labeled_anomalies.csv (optional; searched from DATA_DIR)
    TAD_SMAP_MAX_CHANNELS -> limit the run (default: all SMAP channels)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from telemetry_anomdet.evaluation import best_point_adjusted_f1, windows_to_point_scores
from telemetry_anomdet.feature_extraction.features import make_feature_table
from telemetry_anomdet.ingest import anomaly_point_mask, load_smap, load_smap_labels
from telemetry_anomdet.models.ensemble import AnomalyEnsemble
from telemetry_anomdet.models.unsupervised import KMeansAnomaly, PCAAnomaly
from telemetry_anomdet.preprocessing import pipeline

# A single telemetry channel is a small, uniform feature space, so PCA and
# KMeans emit benign warnings (near-zero variance, fewer clusters than asked).
# Detection is unaffected; quiet them for readable benchmark output.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_DIR = Path(os.environ.get("TAD_SMAP_DIR", "")).expanduser()
LABELS_ENV = os.environ.get("TAD_SMAP_LABELS", "")
MAX_CHANNELS = int(os.environ.get("TAD_SMAP_MAX_CHANNELS", "0"))  # 0 = all

WINDOW_SIZE = 50
STEP = 10


def find_labels_csv() -> Path | None:
    if LABELS_ENV:
        p = Path(LABELS_ENV).expanduser()
        return p if p.exists() else None
    for base in (DATA_DIR, *DATA_DIR.parents[:3]):
        candidate = base / "labeled_anomalies.csv"
        if candidate.exists():
            return candidate
    return None


def score_channel(chan_id: str, sequences) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (point_scores, point_truth) for one channel, or None if unusable."""
    train = pipeline(
        load_smap(DATA_DIR, [chan_id], split="train", dims="telemetry").to_pandas(),
        resample_rule=None,
    )
    test_df = load_smap(DATA_DIR, [chan_id], split="test", dims="telemetry").to_pandas()
    n_points = test_df["timestamp"].nunique()
    test = pipeline(test_df, resample_rule=None)

    X_train = make_feature_table(train, window_size=WINDOW_SIZE, step=STEP)
    X_test = make_feature_table(test, window_size=WINDOW_SIZE, step=STEP)
    if X_train.size == 0 or X_test.size == 0:
        return None

    ensemble = AnomalyEnsemble(
        models={
            "pca": PCAAnomaly(n_components=3, scale=True, percentile=95.0),
            "kmeans": KMeansAnomaly(n_clusters=8, scale=True, percentile=95.0),
        },
        combine="mean",
        normalize="robust",
        percentile=90.0,
    )
    ensemble.fit(X_train)
    win_scores = ensemble.decision_function(X_test)

    scores = windows_to_point_scores(win_scores, n_points, window_size=WINDOW_SIZE, step=STEP)
    truth = anomaly_point_mask(sequences, n_points)
    return scores, truth


def main() -> None:
    labels_csv = find_labels_csv()
    if not DATA_DIR or not (DATA_DIR / "test").exists() or labels_csv is None:
        raise SystemExit(
            "SMAP dataset not found. Set TAD_SMAP_DIR (and optionally "
            "TAD_SMAP_LABELS). See examples/smap_demo.py for the layout.\n"
            f"DATA_DIR: {DATA_DIR or '(unset)'}\nlabels: {labels_csv or '(not found)'}"
        )

    labels = load_smap_labels(labels_csv, spacecraft="SMAP")
    labels = labels.sort_values("anomaly_span", ascending=False).reset_index(drop=True)
    if MAX_CHANNELS > 0:
        labels = labels.head(MAX_CHANNELS)

    print(f"Benchmarking {len(labels)} SMAP channels (window={WINDOW_SIZE}, step={STEP})")
    print("Metric: best-threshold point-adjusted F1 (standard SMAP protocol)\n")
    print(f"{'channel':>8}  {'precision':>9}  {'recall':>6}  {'F1':>6}")
    print("-" * 36)

    all_scores: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    per_channel_f1: list[float] = []
    for _, row in labels.iterrows():
        result = score_channel(row["chan_id"], row["sequences"])
        if result is None:
            continue
        scores, truth = result
        all_scores.append(scores)
        all_truth.append(truth)
        b = best_point_adjusted_f1(scores, truth)
        per_channel_f1.append(b["f1"])
        print(f"{row['chan_id']:>8}  {b['precision']:9.3f}  {b['recall']:6.3f}  {b['f1']:6.3f}")

    if not all_scores:
        raise SystemExit("No channels produced results.")

    scores = np.concatenate(all_scores)
    truth = np.concatenate(all_truth)
    overall = best_point_adjusted_f1(scores, truth, n_thresholds=300)

    print("-" * 36)
    print(
        f"{'GLOBAL':>8}  {overall['precision']:9.3f}  {overall['recall']:6.3f}  {overall['f1']:6.3f}"
    )
    print(
        f"\nAggregate over {len(all_scores)} channels, {len(truth)} points:"
        f"\n  best-threshold point-adjusted F1 (single global threshold): {overall['f1']:.3f}"
        f"  (P={overall['precision']:.3f} R={overall['recall']:.3f})"
        f"\n  per-channel best F1 (oracle per-channel threshold, upper bound): "
        f"{np.mean(per_channel_f1):.3f}"
        f"\n\nNote: the threshold is chosen against the labels (standard SMAP 'best F1'"
        f"\nprotocol). Report as an oracle upper bound, not a deployed operating point."
    )


if __name__ == "__main__":
    main()
