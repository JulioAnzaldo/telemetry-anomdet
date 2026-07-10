"""
SMAP anomaly detection demo.

End to end example: load a SMAP channel with the ingest loader, preprocess it,
window it, fit the classical ensemble on nominal telemetry, and score the test
split against the labeled anomalies.

This uses the public API only:
    telemetry_anomdet.ingest                      -> load_smap, load_smap_labels, anomaly_point_mask
    telemetry_anomdet.preprocessing               -> pipeline
    telemetry_anomdet.feature_extraction.features -> make_feature_table
    telemetry_anomdet.models                      -> PCAAnomaly, KMeansAnomaly, AnomalyEnsemble

Point it at a local copy of the SMAP dataset (telemanom format):
    <data_dir>/train/<chan>.npy, <data_dir>/test/<chan>.npy, labeled_anomalies.csv

Set the location with the TAD_SMAP_DIR environment variable, or edit DATA_DIR below.
The dataset is not shipped with the package. Download it from the telemanom release:
    https://github.com/khundman/telemanom
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from telemetry_anomdet.feature_extraction.features import make_feature_table
from telemetry_anomdet.ingest import anomaly_point_mask, load_smap, load_smap_labels
from telemetry_anomdet.models.ensemble import AnomalyEnsemble
from telemetry_anomdet.models.unsupervised import KMeansAnomaly, PCAAnomaly
from telemetry_anomdet.preprocessing import pipeline

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Directory holding the train/ and test/ subfolders of .npy channel files.
# For the telemanom release this is usually ".../data" (or ".../data/data" if
# the archive was unzipped one level deep).
DATA_DIR = Path(os.environ.get("TAD_SMAP_DIR", "")).expanduser()

# labeled_anomalies.csv typically lives a level or two above DATA_DIR. Set it
# explicitly with TAD_SMAP_LABELS, otherwise we search DATA_DIR and its parents.
LABELS_ENV = os.environ.get("TAD_SMAP_LABELS", "")

WINDOW_SIZE = 50  # samples per window
STEP = 10  # stride between windows
TOP_K = 1  # how many of the most-anomalous channels to demo


def find_labels_csv() -> Path | None:
    """Locate labeled_anomalies.csv from the env var or by walking up from DATA_DIR."""

    if LABELS_ENV:
        p = Path(LABELS_ENV).expanduser()
        return p if p.exists() else None
    for base in (DATA_DIR, *DATA_DIR.parents[:3]):
        candidate = base / "labeled_anomalies.csv"
        if candidate.exists():
            return candidate
    return None


def window_labels(sequences, n_rows: int, window_size: int, step: int) -> np.ndarray:
    """
    Turn point-level anomaly ranges into window-level labels.

    A window is labeled anomalous if it overlaps any anomalous timestep. This
    mirrors how windowify slices the series: window i spans rows
    [i*step, i*step + window_size).
    """

    point_mask = anomaly_point_mask(sequences, n_rows)
    n_windows = max(0, (n_rows - window_size) // step + 1)
    starts = np.arange(n_windows) * step
    return np.array([point_mask[s : s + window_size].any() for s in starts], dtype=bool)


def _windowed(chan_id: str, split: str) -> np.ndarray:
    """Load one channel/split, preprocess, and window it into a 3D tensor.

    dims='telemetry' keeps only the telemetry value (column 0), so the train and
    test splits always share the same feature dimension. resample_rule=None keeps
    the 1:1 timestep alignment needed to line labels up with windows.
    """

    df = load_smap(DATA_DIR, [chan_id], split=split, dims="telemetry").to_pandas()
    clean = pipeline(df, resample_rule=None)
    return make_feature_table(clean, window_size=WINDOW_SIZE, step=STEP)


def run_channel(chan_id: str, sequences) -> None:
    """Train on the nominal split, score the test split, print detection metrics."""

    print(f"\n=== channel {chan_id} ===")

    # Train on nominal telemetry only; score the (anomaly-bearing) test split.
    X_train = _windowed(chan_id, "train")
    X_test = _windowed(chan_id, "test")
    n_rows = (
        load_smap(DATA_DIR, [chan_id], split="test", dims="telemetry")
        .to_pandas()["timestamp"]
        .nunique()
    )

    if X_train.size == 0 or X_test.size == 0:
        print("  not enough data for a single window, skipping")
        return
    print(f"  train windows: {X_train.shape[0]} | test windows: {X_test.shape[0]}")

    # Detectors flatten the 3D windows internally via statistical features.
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
    flags = ensemble.is_anomaly(X_test)

    # Compare against the labeled anomaly windows on the test split.
    truth = window_labels(sequences, n_rows, WINDOW_SIZE, STEP)[: len(flags)]
    tp = int(np.sum(flags & truth))
    fp = int(np.sum(flags & ~truth))
    fn = int(np.sum(~flags & truth))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print(
        f"  windows: {len(flags)} | anomalous (true): {int(truth.sum())} | flagged: {int(flags.sum())}"
    )
    print(f"  precision: {precision:.3f}   recall: {recall:.3f}   (TP={tp} FP={fp} FN={fn})")


def main() -> None:
    labels_csv = find_labels_csv()
    if not DATA_DIR or not (DATA_DIR / "test").exists() or labels_csv is None:
        raise SystemExit(
            "SMAP dataset not found.\n"
            "Set TAD_SMAP_DIR to the directory containing the train/ and test/ "
            "folders of .npy files, and (if needed) TAD_SMAP_LABELS to "
            "labeled_anomalies.csv.\n"
            f"DATA_DIR: {DATA_DIR or '(unset)'}\n"
            f"labels:   {labels_csv or '(not found)'}"
        )

    labels = load_smap_labels(labels_csv, spacecraft="SMAP")
    top = labels.sort_values("anomaly_span", ascending=False).head(TOP_K)
    print(f"Top {TOP_K} SMAP channel(s) by anomaly span: {top['chan_id'].tolist()}")

    for _, row in top.iterrows():
        run_channel(row["chan_id"], row["sequences"])


if __name__ == "__main__":
    main()