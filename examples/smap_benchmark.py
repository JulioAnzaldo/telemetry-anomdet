"""
SMAP detector benchmark: classical baselines vs the graph detectors.

Reports two families of metric, because they disagree sharply and only one of
them describes something deployable.

**Event level** is the headline, and matches how telemanom scores its published
SMAP results: a labelled anomaly counts once if any prediction overlaps it, and
each prediction overlapping nothing counts once against precision. The threshold
behind it is chosen without labels. These numbers are comparable with the
literature and with what an operator would experience.

**Point level** (global and per-channel F1, PR-AUC, false alarm rate) is kept for
continuity. Point-adjusted F1 in particular credits an entire labelled segment to
a single flagged sample and picks its threshold against the labels, so on SMAP an
uninformative detector scores highly; the random@all row exists to show it.

The configurations below are run so the comparison is honest (see the ``dims``
note), with ``random@all`` present as the floor every other row is read against:

    random@all           Uniform random scores. Not a detector, but the floor
                         every other row has to clear. Under point-adjusted F1 it
                         outscores trained configurations, which is the clearest
                         demonstration of why the event-level columns lead.
    classical@telemetry  PCA + KMeans ensemble on the telemetry column only.
                         This is the original classical baseline; its numbers are
                         unchanged from prior releases.
    classical@all        Same ensemble, but on telemetry + all command one-hots
                         (the same multivariate input the graph detectors see). A
                         same-input control so the comparison is apples to apples.
    gdn@all              The Graph Deviation Network on the multivariate input.
                         GDN needs multiple channels to build its sensor graph, so
                         the telemetry-only column would defeat its purpose.
    gdn@telemetry        GDN with no graph and no command context, the same
                         control as kangdn@telemetry. Run both to separate what
                         the KAN layers change from what the graph changes; with
                         only kangdn@telemetry the two are confounded.
    kangdn@all           Same graph and scoring as gdn@all, with KAN layers in
                         place of the ReLU activation and MLP head. Run this to
                         size the KAN configuration: its spline coefficients
                         dominate the distilled artifact, and the count grows as
                         ``embed_dim**2 * (grid_size + spline_order)``, so the
                         smallest configuration that holds its F1 is the one worth
                         exporting.
    kangdn@telemetry     The same detector on the telemetry dimension alone, so
                         there is no graph and no command context. This is a
                         control that isolates what the graph contributes on this
                         dataset, not a recommended configuration: a SMAP record
                         holds one real sensor and 24 command flags, so there is
                         little for a graph to relate. The inter-sensor
                         relationships it exists to capture are what contextual
                         anomalies turn on, and those need a genuinely
                         multi-sensor system to show up.
    kanfeat@*            kangdn with a KAN node feature transform instead of the
                         single linear layer that otherwise carries raw telemetry
                         into the network. Nonlinearity on each node's own
                         history, with none of it discarded.
    arkan@*              AR-KAN as published (Wu et al. 2025): a frozen
                         Yule-Walker filter over the window, then a KAN.
    arkanres@*           Linear(x) + KAN(a * x), so the unfiltered window always
                         reaches the encoder.

Those three vary only the node feature transform, so read them against kangdn@
at matched dims.

Do not read a general conclusion about the feature transforms out of this
benchmark alone. A SMAP record is one real sensor plus 24 near-constant command
flags, so even the @all rows give the graph almost nothing to relate, and the
ranking here inverts on a dataset whose channels are genuinely related. The
measurements across all three datasets, and what they imply for choosing a mode,
are in docs/source/user_guide/feature_transforms.rst.

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
    TAD_SMAP_MAX_CHANNELS -> limit the run (default: all channels)
    TAD_SPACECRAFT        -> SMAP (default) or MSL; both ship in one release
    TAD_GDN_EPOCHS        -> GDN training epochs per channel (default: 30)
    TAD_GDN_DEVICE        -> torch device for GDN and KANGDN (default: auto)
    TAD_KANGDN_EMBED_DIM  -> KANGDN embedding width (default: 64)
    TAD_KANGDN_GRID_SIZE  -> KANGDN spline grid intervals (default: 5)
    TAD_KANGDN_AR_ORDER   -> AR order for the arkan rows (default: full window)
    TAD_BENCH_CONFIGS     -> comma-separated configuration names (default: all)
    TAD_BENCH_SCORES_DIR  -> directory to cache per-channel point scores into
    TAD_BENCH_FROM_CACHE  -> set to 1 to re-score from the cache without training
    TAD_THRESHOLD         -> event-level operating point: dynamic (default) or budget
    TAD_ALARM_BUDGET      -> fraction of points flagged when TAD_THRESHOLD=budget
    TAD_PROTOCOL          -> set to 0 to skip telemanom's sequence filters
    TAD_SCORE_CHANNELS    -> channels allowed to raise an alarm, e.g. 0 for telemetry
    TAD_BENCH_VERBOSE     -> set to 1 to print per-channel rows (default: summary only)
    TAD_WANDB_PROJECT     -> log the run to this Weights & Biases project (off by default)
    TAD_WANDB_ENTITY      -> W&B team or user (optional)
    TAD_WANDB_NAME        -> run name (optional)
    TAD_WANDB_TAGS        -> comma-separated run tags (optional)

Tracking is opt-in and needs the track extra (``uv sync --extra track``). With
TAD_WANDB_PROJECT unset the benchmark behaves and prints exactly as before.

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

from telemetry_anomdet.evaluation import (
    best_point_adjusted_f1,
    evaluate_sequences,
    false_alarm_rate_at_recall,
    pr_auc,
    sequence_prf,
    windows_to_point_scores,
)
from telemetry_anomdet.feature_extraction.features import make_feature_table
from telemetry_anomdet.ingest import anomaly_point_mask, load_smap, load_smap_labels
from telemetry_anomdet.models.ensemble import AnomalyEnsemble
from telemetry_anomdet.models.unsupervised import KMeansAnomaly, PCAAnomaly
from telemetry_anomdet.preprocessing import pipeline
from telemetry_anomdet.thresholding import (
    anomalous_sequences,
    dynamic_threshold,
    filter_sequences,
    startup_skip,
    threshold_for_budget,
)

# A single telemetry channel is a small, uniform feature space, so PCA and
# KMeans emit benign warnings (near zero variance, fewer clusters than asked).
# Detection is unaffected; quiet them for readable benchmark output.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_DIR = Path(os.environ.get("TAD_SMAP_DIR", "")).expanduser()
LABELS_ENV = os.environ.get("TAD_SMAP_LABELS", "")
MAX_CHANNELS = int(os.environ.get("TAD_SMAP_MAX_CHANNELS", "0"))  # 0 = all
# SMAP and MSL ship in one release and share a layout, a labels file and a
# loader; only the spacecraft column separates them. MSL therefore needs no new
# code, just a different filter.
SPACECRAFT = os.environ.get("TAD_SPACECRAFT", "SMAP").upper()
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
# AR order for the arkan rows. Empty means the full context length, one
# coefficient per timestep. A shorter order is better conditioned (the top lags
# of a full-length fit rest on very few sample pairs) and zeroes the oldest lags.
KANGDN_AR_ORDER = (
    int(os.environ["TAD_KANGDN_AR_ORDER"]) if os.environ.get("TAD_KANGDN_AR_ORDER") else None
)

# EWMA factor applied to the per-node forecast errors; empty disables smoothing.
_SMOOTH_ENV = os.environ.get("TAD_SMOOTHING", "").strip()
SMOOTHING = float(_SMOOTH_ENV) if _SMOOTH_ENV else None

# Training seed. Results vary run to run, so a sizing comparison needs repeats
# rather than a single draw per configuration.
SEED = int(os.environ.get("TAD_SEED", "0"))

# Operating point for event-level scoring: "dynamic" selects a threshold from
# the error signal, "budget" caps the fraction of points flagged.
THRESHOLD_METHOD = os.environ.get("TAD_THRESHOLD", "dynamic")
# Apply telemanom's sequence filters so event-level numbers are comparable.
PROTOCOL = os.environ.get("TAD_PROTOCOL", "1") == "1"

# Which feature channels may raise an alarm. A SMAP record is one telemetry
# dimension plus 24 command one-hots, and the labels describe the telemetry, so
# scoring the maximum over all 25 alarms on every command switch. Empty means
# every channel, matching the detector default.
_SC_ENV = os.environ.get("TAD_SCORE_CHANNELS", "").strip()
SCORE_CHANNELS = [int(c) for c in _SC_ENV.split(",")] if _SC_ENV else None
ALARM_BUDGET = float(os.environ.get("TAD_ALARM_BUDGET", "0.05"))

# Recall at which the false alarm rate is reported.
TARGET_RECALL = float(os.environ.get("TAD_TARGET_RECALL", "0.8"))

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
        random_state=SEED,
        smoothing=SMOOTHING,
        score_channels=SCORE_CHANNELS,
    )


class RandomScorer:
    """
    Uniform random scores, as a floor for every other row.

    Point-adjusted F1 rewards touching an anomaly segment anywhere inside it, and
    SMAP's segments are long, so an uninformative detector scores far above zero.
    Without this row there is no way to tell which reported numbers reflect
    detection and which reflect the metric.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X):
        return self

    def decision_function(self, X):
        return np.random.default_rng(self.seed).random(X.shape[0])


def make_random():
    """The uninformative baseline."""
    return RandomScorer(seed=SEED)


def make_kangdn(feat_mode: str = "linear"):
    """
    KANGDN detector: GDN's graph and scoring with KAN layers as the nonlinearities.

    Shares GDN's training hyperparameters so the two rows differ only in
    architecture. ``embed_dim`` and ``grid_size`` are the two knobs that set the
    distilled artifact's size.

    ``feat_mode`` selects the node feature transform. It is the only difference
    between the kangdn, kanfeat, arkan and arkanres rows, so reading those at
    matched dims isolates what the input path is worth, and separates the
    nonlinearity from the AR filter rather than confounding the two.
    """
    from telemetry_anomdet.models.deep import KANGDN

    return KANGDN(
        embed_dim=KANGDN_EMBED_DIM,
        topk=GDN_TOPK,
        lr=GDN_LR,
        epochs=GDN_EPOCHS,
        device=GDN_DEVICE,
        random_state=SEED,
        grid_size=KANGDN_GRID_SIZE,
        spline_order=KANGDN_SPLINE_ORDER,
        smoothing=SMOOTHING,
        score_channels=SCORE_CHANNELS,
        feat_mode=feat_mode,
        ar_order=KANGDN_AR_ORDER,
    )


def make_kanfeat():
    """KAN feature transform, no AR filter: the nonlinearity on its own."""
    return make_kangdn(feat_mode="kan")


def make_arkan():
    """AR-KAN as published: frozen Yule-Walker filter, then a KAN."""
    return make_kangdn(feat_mode="ar_kan")


def make_arkanres():
    """AR-KAN as a residual on a linear branch, so the window is never lost."""
    return make_kangdn(feat_mode="ar_kan_residual")


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
    ("random@all", "all", make_random, GDN_WINDOW),
    ("classical@telemetry", "telemetry", make_classical, WINDOW_SIZE),
    ("classical@all", "all", make_classical, WINDOW_SIZE),
    ("gdn@all", "all", make_gdn, GDN_WINDOW),
    ("gdn@telemetry", "telemetry", make_gdn, GDN_WINDOW),
    ("kangdn@all", "all", make_kangdn, GDN_WINDOW),
    ("kangdn@telemetry", "telemetry", make_kangdn, GDN_WINDOW),
    ("kanfeat@all", "all", make_kanfeat, GDN_WINDOW),
    ("kanfeat@telemetry", "telemetry", make_kanfeat, GDN_WINDOW),
    ("arkan@all", "all", make_arkan, GDN_WINDOW),
    ("arkan@telemetry", "telemetry", make_arkan, GDN_WINDOW),
    ("arkanres@all", "all", make_arkanres, GDN_WINDOW),
    ("arkanres@telemetry", "telemetry", make_arkanres, GDN_WINDOW),
]

# Configurations that need torch, so a missing deep extra skips them by name
# rather than by prefix matching.
NEEDS_TORCH = frozenset(
    name
    for name, *_ in CONFIGS
    if name.split("@", 1)[0] in {"gdn", "kangdn", "kanfeat", "arkan", "arkanres"}
)


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

    ``spacecraft`` is part of the key even though a cache entry is stored per
    channel. SMAP and MSL happen to use disjoint chan_ids in telemanom's
    labeled_anomalies.csv, so without it their entries coexist in one directory
    rather than overwriting: a run reads only its own channels and the scores
    stay correct, but the directory holds both missions and any later dataset
    that reused an id would silently serve the wrong mission's scores. Keying on
    it makes the separation a property of the cache rather than a coincidence.
    """
    params = {
        "config": name,
        "spacecraft": SPACECRAFT,
        "window_size": window_size,
        "step": STEP,
        "seed": SEED,
        "smoothing": SMOOTHING,
        "score_channels": SCORE_CHANNELS,
    }
    # Matched on the model prefix, not the full name: the @telemetry rows are
    # swept over exactly the same hyperparameters as the @all rows, so keying
    # only the latter let two sizing runs at different embed_dim collide on one
    # cache entry and silently return the first run's scores.
    model = name.split("@", 1)[0]
    if model == "gdn":
        params.update(embed_dim=GDN_EMBED_DIM, topk=GDN_TOPK, lr=GDN_LR, epochs=GDN_EPOCHS)
    elif model in ("kangdn", "kanfeat", "arkan", "arkanres"):
        params.update(
            embed_dim=KANGDN_EMBED_DIM,
            grid_size=KANGDN_GRID_SIZE,
            spline_order=KANGDN_SPLINE_ORDER,
            topk=GDN_TOPK,
            lr=GDN_LR,
            epochs=GDN_EPOCHS,
        )
    if model in ("arkan", "arkanres"):
        params.update(ar_order=KANGDN_AR_ORDER)
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
    event_rows: list[dict] = []
    by_class: dict[str, list[int]] = {}
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

        # Event-level scoring needs a single operating point rather than a
        # sweep, chosen without labels so the result is deployable.
        if THRESHOLD_METHOD == "budget":
            cut = threshold_for_budget(scores, ALARM_BUDGET)["threshold"]
        else:
            cut = dynamic_threshold(scores)["threshold"]
        predicted = anomalous_sequences(scores, cut)
        if PROTOCOL:
            # telemanom drops single-sample runs and ignores the cold start,
            # where the model has no history yet.
            predicted = filter_sequences(
                predicted,
                min_length=2,
                ignore_before=startup_skip(scores.size, window_size),
            )
        event_rows.append(
            evaluate_sequences(predicted, anomalous_sequences(truth.astype(float), 0.5))
        )

        # Point and contextual anomalies behave differently under a forecasting
        # detector, so a single recall hides which kind is being missed.
        for (a, b_end), label in zip(row["sequences"], row["classes"], strict=False):
            if a >= scores.size:
                continue
            end = min(b_end, scores.size - 1)
            hit = any(not (y < a or x > end) for x, y in predicted)
            by_class.setdefault(label, [0, 0])[0 if hit else 1] += 1

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

    # Threshold-free and operating-point metrics. Point-adjusted F1 is retained
    # for comparability with published SMAP results, but it is inflated by the
    # adjustment, so it is reported beside metrics that are not.
    events = sequence_prf(event_rows)
    overall["event"] = events
    overall["by_class"] = by_class
    overall["pr_auc"] = pr_auc(scores, truth)
    overall["base_rate"] = float(truth.mean())
    fa = false_alarm_rate_at_recall(scores, truth, target_recall=TARGET_RECALL)
    overall["false_alarm_rate"] = fa["false_alarm_rate"]

    print(
        f"  global best-F1: {overall['f1']:.3f}  "
        f"(P={overall['precision']:.3f} R={overall['recall']:.3f})   "
        f"per-channel mean best-F1: {overall['per_channel_f1']:.3f}   "
        f"[{overall['n_channels']} channels]"
    )
    print(
        f"  PR-AUC: {overall['pr_auc']:.3f} (random = {overall['base_rate']:.3f})   "
        f"false alarms at R={TARGET_RECALL:.2f}: {overall['false_alarm_rate']:.3f}"
    )
    print(
        f"  event-level ({THRESHOLD_METHOD}): P {events['precision']:.3f} "
        f"R {events['recall']:.3f} F1 {events['f1']:.3f} F0.5 {events['f_half']:.3f}   "
        f"TP {events['true_positives']} FP {events['false_positives']} "
        f"FN {events['false_negatives']}"
    )
    if by_class:
        parts = [
            f"{name} {hit}/{hit + miss} ({hit / max(hit + miss, 1):.2f})"
            for name, (hit, miss) in sorted(by_class.items())
        ]
        print(f"  recall by anomaly class: {'   '.join(parts)}")
    return overall


# ---------------------------------------------------------------------------
# Experiment tracking (optional)
# ---------------------------------------------------------------------------
#
# Off unless TAD_WANDB_PROJECT is set, and wandb is an optional extra, so the
# benchmark's behaviour and output are identical without it. Nothing in the
# library imports wandb; only this script does.
#
# One run per invocation, with each configuration's metrics under its own
# prefix. A sweep sets TAD_BENCH_CONFIGS to a single name, so a sweep trial is
# one run with one prefix, and a manual comparison is one run with several.

WANDB_PROJECT = os.environ.get("TAD_WANDB_PROJECT", "")


def wandb_run_config(labels) -> dict:
    """
    Everything that could change a number, recorded with the run.

    Every TAD_* variable is captured rather than a hand-picked few: a result
    that cannot be traced back to the settings that produced it is not
    reproducible, and the one setting nobody thought to record is the one that
    explains the discrepancy.
    """
    config = {key: value for key, value in sorted(os.environ.items()) if key.startswith("TAD_")}
    config.update(
        {
            "spacecraft": SPACECRAFT,
            "n_channels": len(labels),
            "channels": ",".join(labels["chan_id"]),
            "threshold_method": THRESHOLD_METHOD,
            "seed": SEED,
            "step": STEP,
            "gdn_epochs": GDN_EPOCHS,
            "gdn_embed_dim": GDN_EMBED_DIM,
            "gdn_topk": GDN_TOPK,
            "kangdn_embed_dim": KANGDN_EMBED_DIM,
            "kangdn_grid_size": KANGDN_GRID_SIZE,
            "smoothing": SMOOTHING,
            "score_channels": str(SCORE_CHANNELS),
            "from_cache": FROM_CACHE,
        }
    )
    return config


def wandb_start(labels):
    """Begin a run, or return None when tracking is off or wandb is absent."""
    if not WANDB_PROJECT:
        return None
    try:
        import wandb
    except ImportError:
        print("  (TAD_WANDB_PROJECT is set but wandb is not installed; skipping tracking)")
        print(
            '   install it with: uv sync --extra track  or  pip install "telemetry-anomdet[track]"'
        )
        return None

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=os.environ.get("TAD_WANDB_ENTITY") or None,
        name=os.environ.get("TAD_WANDB_NAME") or None,
        tags=[t for t in os.environ.get("TAD_WANDB_TAGS", "").split(",") if t],
        config=wandb_run_config(labels),
    )
    # run.url is None when WANDB_MODE=offline, which is a normal way to record
    # a run on a machine with no network and sync it later.
    print(f"  tracking to wandb: {run.url or f'offline, sync later from {run.dir}'}")
    return run


def wandb_log_config(run, name: str, overall: dict) -> None:
    """Log one configuration's aggregate metrics under its own prefix."""
    if run is None:
        return
    events = overall["event"]
    metrics = {
        # Event level first: these are the deployable numbers and the ones a
        # sweep should ever be pointed at.
        f"{name}/event_f1": events["f1"],
        f"{name}/event_f_half": events["f_half"],
        f"{name}/event_precision": events["precision"],
        f"{name}/event_recall": events["recall"],
        f"{name}/event_tp": events["true_positives"],
        f"{name}/event_fp": events["false_positives"],
        f"{name}/event_fn": events["false_negatives"],
        # Point level, kept for continuity and inflated by the adjustment.
        f"{name}/point_adjusted_f1": overall["f1"],
        f"{name}/per_channel_f1": overall["per_channel_f1"],
        f"{name}/pr_auc": overall["pr_auc"],
        f"{name}/false_alarm_rate": overall["false_alarm_rate"],
        f"{name}/n_channels": overall["n_channels"],
    }
    for label, (hit, miss) in overall.get("by_class", {}).items():
        total = hit + miss
        if total:
            metrics[f"{name}/recall_{label}"] = hit / total
    run.log(metrics)


def wandb_log_summary(run, results: dict) -> None:
    """Log the side-by-side table, and the event F1 of each configuration."""
    if run is None:
        return
    import wandb

    table = wandb.Table(
        columns=[
            "configuration",
            "event_f1",
            "event_precision",
            "event_recall",
            "tp",
            "fp",
            "fn",
            "point_adjusted_f1",
            "pr_auc",
            "false_alarm_rate",
        ]
    )
    for name, r in results.items():
        e = r["event"]
        table.add_data(
            name,
            e["f1"],
            e["precision"],
            e["recall"],
            e["true_positives"],
            e["false_positives"],
            e["false_negatives"],
            r["f1"],
            r["pr_auc"],
            r["false_alarm_rate"],
        )
        # Summary values are what a sweep sorts on, so they are event level.
        run.summary[f"{name}/event_f1"] = e["f1"]
    run.log({"summary": table})


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

    labels = load_smap_labels(labels_csv, spacecraft=SPACECRAFT)
    labels = labels.sort_values("anomaly_span", ascending=False).reset_index(drop=True)
    if MAX_CHANNELS > 0:
        labels = labels.head(MAX_CHANNELS)

    configs = select_configs()

    print(f"Benchmarking {len(labels)} {SPACECRAFT} channels (step={STEP})")
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

    run = wandb_start(labels)

    results: dict[str, dict] = {}
    for name, dims, make_detector, window_size in configs:
        if name in NEEDS_TORCH and not TORCH_AVAILABLE and not FROM_CACHE:
            continue
        outcome = run_config(name, dims, make_detector, window_size, labels)
        if outcome is not None:
            results[name] = outcome
            wandb_log_config(run, name, outcome)

    # Final side-by-side comparison.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(
        f"{'configuration':>20}  {'global F1':>9}  {'per-chan F1':>11}  {'PR-AUC':>7}  "
        f"{'FA@R':>6}  {'evP':>6} {'evR':>6} {'evF1':>6} {'evF.5':>6}  "
        f"{'TP':>4} {'FP':>4} {'FN':>4}"
    )
    print("-" * 116)
    for name in results:
        r = results[name]
        e = r["event"]
        print(
            f"{name:>20}  {r['f1']:9.3f}  {r['per_channel_f1']:11.3f}  "
            f"{r['pr_auc']:7.3f}  {r['false_alarm_rate']:6.3f}  "
            f"{e['precision']:6.3f} {e['recall']:6.3f} {e['f1']:6.3f} {e['f_half']:6.3f}  "
            f"{e['true_positives']:4d} {e['false_positives']:4d} {e['false_negatives']:4d}"
        )
    if results:
        floor = next(iter(results.values()))["base_rate"]
        print(f"{'PR-AUC floor (random)':>20}  {'':>9}  {'':>11}  {floor:7.3f}")
    print(
        "\nevP/evR/evF1 are event-level, scored as telemanom scores its published"
        "\nSMAP results: a labelled anomaly counts once if any prediction overlaps it,"
        "\nand each prediction overlapping nothing counts once against precision. The"
        "\nthreshold behind them is chosen without labels, so those columns describe a"
        "\ndeployable operating point and are the ones to compare against published"
        "\nnumbers. For reference telemanom reports P 0.838 R 0.899 F1 0.867 on SMAP,"
        "\nfrom TP 62 FP 12 FN 7 over 69 labelled anomalies in the same 54 channels."
        "\n"
        "\nThe remaining columns are point-level and much more forgiving. Global F1"
        "\nmarks a whole segment detected from one flagged point and picks its"
        "\nthreshold against the labels, so on SMAP's long segments even the random@all"
        "\nrow scores highly. PR-AUC is threshold free and sits at the base rate for"
        "\nrandom scores. FA@R is the false alarm rate at the target recall."
    )

    if run is not None:
        wandb_log_summary(run, results)
        run.finish()


if __name__ == "__main__":
    main()
