"""
SMAP detector benchmark: classical baselines vs GDN.

Reports point adjusted F1. For each channel it trains on the nominal train split 
and scores the test split, sweeps thresholds, and reports the best point-adjusted
F1. Because the threshold is selected against the labels, treat every number as 
an oracle threshold upper bound, not a deployable operating point.

Three configurations are run so the comparison is honest (see the ``dims`` note):

    classical@telemetry  PCA + KMeans ensemble on the telemetry column only.
                         This is the original classical baseline; its numbers are
                         unchanged from prior releases.
    classical@all        Same ensemble, but on telemetry + all command one-hots
                         (the same multivariate input GDN sees). A same-input
                         control so the GDN comparison is apples to apples.
    gdn@all              The Graph Deviation Network on the multivariate input.
                         GDN needs multiple channels to build its sensor graph, so
                         the telemetry-only column would defeat its purpose.

Why ``all`` and not ``nonzero``? ``nonzero`` drops all-zero columns per split, so
train and test can end up with different feature counts (a command inactive in
one split but not the other) and the fitted scaler/graph stop aligning. ``all``
pins a fixed schema across splits.

Why keep telemetry-only as its own row? The command one-hots are near constant
binary flags; pushing them through the classical detectors' statistical features
injects near zero variance noise and handicaps them. Telemetry-only is the
classical detectors honest best input, so it is kept rather than replaced.

Point it at a local copy of the SMAP dataset (telemanom format), same as
examples/smap_demo.py:
    TAD_SMAP_DIR          -> directory containing train/ and test/ .npy files
    TAD_SMAP_LABELS       -> labeled_anomalies.csv (optional; searched from DATA_DIR)
    TAD_SMAP_MAX_CHANNELS -> limit the run (default: all SMAP channels)
    TAD_GDN_EPOCHS        -> GDN training epochs per channel (default: 30)
    TAD_GDN_DEVICE        -> torch device for GDN, e.g. "cuda" (default: auto)
    TAD_BENCH_VERBOSE     -> set to 1 to print per-channel rows (default: summary only)

GDN requires the optional deep extra (torch). If torch is not installed the
gdn@nonzero configuration is skipped with a note; the classical rows still run.
"""

from __future__ import annotations

import importlib.util
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
# KMeans emit benign warnings (near zero variance, fewer clusters than asked).
# Detection is unaffected; quiet them for readable benchmark output.
warnings.filterwarnings("ignore", category = RuntimeWarning)
warnings.filterwarnings("ignore", category = ConvergenceWarning)

DATA_DIR = Path(os.environ.get("TAD_SMAP_DIR", "")).expanduser()
LABELS_ENV = os.environ.get("TAD_SMAP_LABELS", "")
MAX_CHANNELS = int(os.environ.get("TAD_SMAP_MAX_CHANNELS", "0"))  # 0 = all
GDN_DEVICE = os.environ.get("TAD_GDN_DEVICE", "") or None
VERBOSE = os.environ.get("TAD_BENCH_VERBOSE", "") == "1"

WINDOW_SIZE = 50
STEP = 10

# GDN hyperparameters, overridable so a sweep's winning config can be replayed in
# one command. Defaults match GDN's own defaults; TAD_GDN_WINDOW defaults to the
# shared WINDOW_SIZE. Only the gdn@all row uses these.
GDN_EPOCHS = int(os.environ.get("TAD_GDN_EPOCHS", "30"))
GDN_EMBED_DIM = int(os.environ.get("TAD_GDN_EMBED_DIM", "64"))
GDN_TOPK = int(os.environ.get("TAD_GDN_TOPK", "15"))
GDN_LR = float(os.environ.get("TAD_GDN_LR", "0.001"))
GDN_WINDOW = int(os.environ.get("TAD_GDN_WINDOW", str(WINDOW_SIZE)))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def find_labels_csv() -> Path | None:
    if LABELS_ENV:
        p = Path(LABELS_ENV).expanduser()
        return p if p.exists() else None
    for base in (DATA_DIR, *DATA_DIR.parents[:3]):
        candidate = base / "labeled_anomalies.csv"
        if candidate.exists():
            return candidate
    return None


def make_classical() -> AnomalyEnsemble:
    """The classical PCA + KMeans ensemble (flattens the 3D windows internally)."""
    return AnomalyEnsemble(
        models={
            "pca": PCAAnomaly(n_components=3, scale=True, percentile=95.0),
            "kmeans": KMeansAnomaly(n_clusters=8, scale=True, percentile=95.0),
        },
        combine="mean",
        normalize="robust",
        percentile=90.0,
    )


def make_gdn():
    """GDN detector (consumes the 3D windows directly; self-normalizing)."""
    from telemetry_anomdet.models.deep import GDN

    return GDN(
        embed_dim=GDN_EMBED_DIM,
        topk=GDN_TOPK,
        lr=GDN_LR,
        epochs=GDN_EPOCHS,
        device=GDN_DEVICE,
        random_state=0,
    )


# name -> (dims, detector factory, window_size). Kept as data so main() iterates.
# window_size is per-config so a sweep can tune it for GDN (via TAD_GDN_WINDOW)
# without disturbing the classical baselines' fixed window.
#
# The multivariate rows use dims = "all" (not "nonzero") on purpose: "nonzero"
# drops all-zero columns *per split*, so a command one-hot inactive in train but
# active in test gives the two splits different feature counts and the fitted
# scaler/graph no longer align. "all" pins a fixed 25-dim schema; constant
# columns are harmless (zero variance -> scale 1.0; GDN just gets a self-loop).
CONFIGS: list[tuple[str, str, object, int]] = [
    ("classical@telemetry", "telemetry", make_classical, WINDOW_SIZE),
    ("classical@all", "all", make_classical, WINDOW_SIZE),
    ("gdn@all", "all", make_gdn, GDN_WINDOW),
]


def build_windows(chan_id: str, dims: str, window_size: int):
    """Load + preprocess one channel at the given dims -> (X_train, X_test, n_points)."""
    train = pipeline(
        load_smap(DATA_DIR, [chan_id], split="train", dims=dims).to_pandas(),
        resample_rule=None,
    )
    test_df = load_smap(DATA_DIR, [chan_id], split="test", dims=dims).to_pandas()
    n_points = test_df["timestamp"].nunique()
    test = pipeline(test_df, resample_rule=None)

    X_train = make_feature_table(train, window_size=window_size, step=STEP)
    X_test = make_feature_table(test, window_size=window_size, step=STEP)
    return X_train, X_test, n_points


def score_channel(chan_id, sequences, dims, make_detector, window_size):
    """Return (point_scores, point_truth) for one channel/config, or None if unusable."""
    X_train, X_test, n_points = build_windows(chan_id, dims, window_size)
    if X_train.size == 0 or X_test.size == 0:
        return None

    detector = make_detector()
    detector.fit(X_train)
    win_scores = detector.decision_function(X_test)

    scores = windows_to_point_scores(win_scores, n_points, window_size=window_size, step=STEP)
    truth = anomaly_point_mask(sequences, n_points)
    return scores, truth


def run_config(name: str, dims: str, make_detector, window_size: int, labels) -> dict | None:
    """Run one configuration across all channels; return its aggregate metrics."""
    print(f"\n=== {name}  (dims={dims}, window={window_size}) ===")
    if VERBOSE:
        print(f"{'channel':>8}  {'precision':>9}  {'recall':>6}  {'F1':>6}")
        print("-" * 36)

    all_scores: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    per_channel_f1: list[float] = []
    for _, row in labels.iterrows():
        result = score_channel(row["chan_id"], row["sequences"], dims, make_detector, window_size)
        if result is None:
            continue
        scores, truth = result
        all_scores.append(scores)
        all_truth.append(truth)
        b = best_point_adjusted_f1(scores, truth)
        per_channel_f1.append(b["f1"])
        if VERBOSE:
            print(f"{row['chan_id']:>8}  {b['precision']:9.3f}  {b['recall']:6.3f}  {b['f1']:6.3f}")

    if not all_scores:
        print("  (no channels produced results)")
        return None

    scores = np.concatenate(all_scores)
    truth = np.concatenate(all_truth)
    overall = best_point_adjusted_f1(scores, truth, n_thresholds=300)
    overall["per_channel_f1"] = float(np.mean(per_channel_f1))
    overall["n_channels"] = len(all_scores)

    print(
        f"  global best-F1: {overall['f1']:.3f}  "
        f"(P={overall['precision']:.3f} R={overall['recall']:.3f})   "
        f"per-channel mean best-F1: {overall['per_channel_f1']:.3f}   "
        f"[{overall['n_channels']} channels]"
    )
    return overall


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

    print(f"Benchmarking {len(labels)} SMAP channels (step={STEP})")
    print("Metric: best-threshold point-adjusted F1 (standard SMAP protocol)")
    if not TORCH_AVAILABLE:
        print(
            "Note: torch not installed -> gdn@all will be skipped. "
            "Install the deep extra to include GDN: uv sync --extra deep"
        )

    results: dict[str, dict] = {}
    for name, dims, make_detector, window_size in CONFIGS:
        if name.startswith("gdn") and not TORCH_AVAILABLE:
            continue
        outcome = run_config(name, dims, make_detector, window_size, labels)
        if outcome is not None:
            results[name] = outcome

    # Final side-by-side comparison.
    print("\n" + "=" * 60)
    print("SUMMARY  (best-threshold point-adjusted F1, oracle upper bound)")
    print(f"{'configuration':>20}  {'global F1':>9}  {'per-chan F1':>11}")
    print("-" * 46)
    for name in results:
        r = results[name]
        print(f"{name:>20}  {r['f1']:9.3f}  {r['per_channel_f1']:11.3f}")
    print(
        "\nNote: thresholds are chosen against the labels (standard SMAP 'best F1'"
        "\nprotocol). Report as an oracle upper bound, not a deployed operating point."
        "\nclassical@all is the same input control for a fair GDN comparison;"
        "\nclassical@telemetry is the classical detectors' native best input."
    )


if __name__ == "__main__":
    main()
