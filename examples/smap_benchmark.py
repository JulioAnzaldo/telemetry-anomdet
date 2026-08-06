"""
SMAP detector benchmark: classical baselines vs the graph detectors.

Reports point adjusted F1. For each channel it trains on the nominal train split
and scores the test split, sweeps thresholds, and reports the best point-adjusted
F1. Because the threshold is selected against the labels, treat every number as
an oracle threshold upper bound, not a deployable operating point.

Four configurations are run so the comparison is honest (see the ``dims`` note):

    classical@telemetry  PCA + KMeans ensemble on the telemetry column only.
                         This is the original classical baseline; its numbers are
                         unchanged from prior releases.
    classical@all        Same ensemble, but on telemetry + all command one-hots
                         (the same multivariate input the graph detectors see). A
                         same-input control so the comparison is apples to apples.
    gdn@all              The Graph Deviation Network on the multivariate input.
                         GDN needs multiple channels to build its sensor graph, so
                         the telemetry-only column would defeat its purpose.
    kangdn@all           Same graph and scoring as gdn@all, with KAN layers in
                         place of the ReLU activation and MLP head. Run this to
                         size the KAN configuration: its spline coefficients
                         dominate the distilled artifact, and the count grows as
                         ``embed_dim**2 * (grid_size + spline_order)``, so the
                         smallest configuration that holds its F1 is the one worth
                         exporting.

Running one row at a time: set ``TAD_BENCH_CONFIGS`` to a comma-separated list of
configuration names. A sizing sweep over the KAN layers therefore looks like::

    TAD_BENCH_CONFIGS=kangdn@all TAD_KANGDN_EMBED_DIM=16 python examples/smap_benchmark.py

Names are matched exactly against the ``CONFIGS`` table below; an unknown name is
an error rather than a silent empty run.

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
    TAD_GDN_DEVICE        -> torch device for GDN and KANGDN (default: auto)
    TAD_KANGDN_EMBED_DIM  -> KANGDN embedding width (default: 64)
    TAD_KANGDN_GRID_SIZE  -> KANGDN spline grid intervals (default: 5)
    TAD_BENCH_CONFIGS     -> comma-separated configuration names (default: all)
    TAD_BENCH_SCORES_DIR  -> directory to cache per-channel point scores into
    TAD_BENCH_FROM_CACHE  -> set to 1 to re-score from the cache without training
    TAD_BENCH_VERBOSE     -> set to 1 to print per-channel rows (default: summary only)

Training and measurement are separable. Point the run at a cache directory once,
and every later metric change is a pass over saved arrays instead of a full
retrain of every channel::

    TAD_BENCH_SCORES_DIR=eval/scores python examples/smap_benchmark.py
    TAD_BENCH_SCORES_DIR=eval/scores TAD_BENCH_FROM_CACHE=1 python examples/smap_benchmark.py

The second form needs neither the .npy dataset nor torch, since a cache entry
holds the point scores and the ground truth for one channel.

GDN and KANGDN require the optional deep extra (torch). If torch is not installed
those configurations are skipped with a note; the classical rows still run.
"""

from __future__ import annotations

import hashlib
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
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

# KANGDN hyperparameters. The training ones default to GDN's so the two rows stay
# comparable and only the architecture differs; embed_dim and grid_size are split
# out because they set the size of the distilled artifact.
KANGDN_EMBED_DIM = int(os.environ.get("TAD_KANGDN_EMBED_DIM", str(GDN_EMBED_DIM)))
KANGDN_GRID_SIZE = int(os.environ.get("TAD_KANGDN_GRID_SIZE", "5"))
KANGDN_SPLINE_ORDER = int(os.environ.get("TAD_KANGDN_SPLINE_ORDER", "3"))

# Configuration names to run; empty means all of them.
SELECTED = [n.strip() for n in os.environ.get("TAD_BENCH_CONFIGS", "").split(",") if n.strip()]

# Score cache. Training a configuration and measuring it are separate concerns:
# with a cache, adding or changing a metric costs a pass over saved arrays rather
# than a full retrain of every channel.
_SCORES_ENV = os.environ.get("TAD_BENCH_SCORES_DIR", "").strip()
SCORES_DIR = Path(_SCORES_ENV).expanduser() if _SCORES_ENV else None
FROM_CACHE = os.environ.get("TAD_BENCH_FROM_CACHE", "") == "1"

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


def make_kangdn():
    """
    KANGDN detector: GDN's graph and scoring with KAN layers as the nonlinearities.

    Shares GDN's training hyperparameters so the two rows differ only in
    architecture. ``embed_dim`` and ``grid_size`` are the two knobs that set the
    distilled artifact's size.
    """
    from telemetry_anomdet.models.deep import KANGDN

    return KANGDN(
        embed_dim=KANGDN_EMBED_DIM,
        topk=GDN_TOPK,
        lr=GDN_LR,
        epochs=GDN_EPOCHS,
        device=GDN_DEVICE,
        random_state=0,
        grid_size=KANGDN_GRID_SIZE,
        spline_order=KANGDN_SPLINE_ORDER,
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
    ("kangdn@all", "all", make_kangdn, GDN_WINDOW),
]

# Configurations that need torch, so a missing deep extra skips them by name
# rather than by prefix matching.
NEEDS_TORCH = frozenset({"gdn@all", "kangdn@all"})


def select_configs() -> list[tuple[str, str, object, int]]:
    """
    Resolve TAD_BENCH_CONFIGS to the subset of CONFIGS to run.

    Returns every configuration when the variable is unset. Raises on an unknown
    name so that a typo fails loudly instead of silently benchmarking nothing.
    """
    if not SELECTED:
        return CONFIGS
    known = {name for name, *_ in CONFIGS}
    unknown = [name for name in SELECTED if name not in known]
    if unknown:
        raise SystemExit(
            f"Unknown configuration(s) in TAD_BENCH_CONFIGS: {', '.join(unknown)}\n"
            f"Available: {', '.join(sorted(known))}"
        )
    return [config for config in CONFIGS if config[0] in SELECTED]


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


def config_params(name: str, window_size: int) -> dict:
    """
    The settings that determine a configuration's scores.

    Two runs sharing a configuration name but differing here (a sizing sweep
    varying embed_dim, say) are different experiments and must not share a cache
    entry.
    """
    params = {"config": name, "window_size": window_size, "step": STEP}
    if name == "gdn@all":
        params.update(embed_dim=GDN_EMBED_DIM, topk=GDN_TOPK, lr=GDN_LR, epochs=GDN_EPOCHS)
    elif name == "kangdn@all":
        params.update(
            embed_dim=KANGDN_EMBED_DIM,
            grid_size=KANGDN_GRID_SIZE,
            spline_order=KANGDN_SPLINE_ORDER,
            topk=GDN_TOPK,
            lr=GDN_LR,
            epochs=GDN_EPOCHS,
        )
    return params


def config_fingerprint(name: str, window_size: int) -> str:
    """Short stable digest of config_params, used to key the cache directory."""
    blob = repr(sorted(config_params(name, window_size).items())).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


def cache_path(name: str, chan_id: str, window_size: int) -> Path:
    """Location of one channel's cached scores for a configuration."""
    tag = f"{name.replace('@', '_at_')}-{config_fingerprint(name, window_size)}"
    return SCORES_DIR / tag / f"{chan_id}.npz"


def save_scores(name, chan_id, window_size, scores, truth) -> None:
    """Persist a channel's point scores, ground truth, and the settings used."""
    path = cache_path(name, chan_id, window_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, scores=scores, truth=truth, params=repr(config_params(name, window_size))
    )
    params_file = path.parent / "params.txt"
    if not params_file.exists():
        params_file.write_text(repr(config_params(name, window_size)), encoding="utf-8")


def load_scores(name: str, chan_id: str, window_size: int):
    """Read a channel's cached scores, or None when it was never written."""
    path = cache_path(name, chan_id, window_size)
    if not path.exists():
        return None
    with np.load(path) as data:
        return data["scores"], data["truth"]


def run_config(name: str, dims: str, make_detector, window_size: int, labels) -> dict | None:
    """
    Run one configuration across all channels; return its aggregate metrics.

    Scores come from training a detector, or from the cache when
    TAD_BENCH_FROM_CACHE is set. Re-scoring from cache needs neither the dataset
    nor torch, so a metric change is seconds rather than a full retrain.
    """
    source = "cache" if FROM_CACHE else "training"
    tag = f", id={config_fingerprint(name, window_size)}" if SCORES_DIR is not None else ""
    print(f"\n=== {name}  (dims={dims}, window={window_size}, from {source}{tag}) ===")
    if VERBOSE:
        print(f"{'channel':>8}  {'precision':>9}  {'recall':>6}  {'F1':>6}")
        print("-" * 36)

    all_scores: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    per_channel_f1: list[float] = []
    for _, row in labels.iterrows():
        if FROM_CACHE:
            result = load_scores(name, row["chan_id"], window_size)
        else:
            result = score_channel(
                row["chan_id"], row["sequences"], dims, make_detector, window_size
            )
            if result is not None and SCORES_DIR is not None:
                save_scores(name, row["chan_id"], window_size, *result)
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
        if FROM_CACHE:
            # An empty cache read almost always means the settings differ from
            # the run that populated it, which changes the fingerprint.
            print(f"  (nothing cached under {cache_path(name, '<channel>', window_size).parent})")
            print("   check that the hyperparameters match the run that wrote the cache")
        else:
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
    if FROM_CACHE and SCORES_DIR is None:
        raise SystemExit("TAD_BENCH_FROM_CACHE=1 requires TAD_BENCH_SCORES_DIR.")

    labels_csv = find_labels_csv()
    # Re-scoring from cache needs the labels file for the channel list, but not
    # the .npy dataset: the point scores and ground truth are already saved.
    if labels_csv is None or (
        not FROM_CACHE and (not DATA_DIR or not (DATA_DIR / "test").exists())
    ):
        raise SystemExit(
            "SMAP dataset not found. Set TAD_SMAP_DIR (and optionally "
            "TAD_SMAP_LABELS). See examples/smap_demo.py for the layout.\n"
            f"DATA_DIR: {DATA_DIR or '(unset)'}\nlabels: {labels_csv or '(not found)'}"
        )

    labels = load_smap_labels(labels_csv, spacecraft="SMAP")
    labels = labels.sort_values("anomaly_span", ascending=False).reset_index(drop=True)
    if MAX_CHANNELS > 0:
        labels = labels.head(MAX_CHANNELS)

    configs = select_configs()

    print(f"Benchmarking {len(labels)} SMAP channels (step={STEP})")
    print("Metric: best-threshold point-adjusted F1 (standard SMAP protocol)")
    print(f"Configurations: {', '.join(name for name, *_ in configs)}")
    if SCORES_DIR is not None:
        print(f"Score cache: {SCORES_DIR}" + ("  (reading)" if FROM_CACHE else "  (writing)"))
    # Cached scores are plain arrays, so re-scoring never needs torch.
    if not TORCH_AVAILABLE and not FROM_CACHE and any(name in NEEDS_TORCH for name, *_ in configs):
        print(
            "Note: torch not installed -> the graph detector rows will be skipped. "
            "Install the deep extra to include them: uv sync --extra deep"
        )

    results: dict[str, dict] = {}
    for name, dims, make_detector, window_size in configs:
        if name in NEEDS_TORCH and not TORCH_AVAILABLE and not FROM_CACHE:
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
