"""
Export a fitted KANGDN as C, with golden vectors, ready to drop into a target port.

Writes ``kangdn.h``, ``kangdn.c`` and ``kangdn_vectors.h`` into an output
directory. The vectors header carries a handful of input windows beside the
reference evaluator's score for each, so a target validates itself: score every
window, compare, report. No host tooling is needed on the device.

Usage::

    python examples/export_kangdn_c.py targets/host/generated

Trains on SMAP when ``TAD_SMAP_DIR`` points at a local copy, and on synthetic
data shaped like a SMAP channel otherwise, which is enough to exercise a port.

    TAD_SMAP_DIR        -> directory containing train/ and test/ .npy files
    TAD_SMAP_CHANNEL    -> channel to train on (default: D-2)
    TAD_EXPORT_DTYPE    -> "float" (default) or "double"
    TAD_EXPORT_VECTORS  -> number of golden vectors to emit (default: 8)
    TAD_KANGDN_EMBED_DIM, TAD_KANGDN_GRID_SIZE, TAD_GDN_EPOCHS, TAD_GDN_WINDOW

The deployed configuration should come from a sizing run rather than these
defaults; see the onboard deployment page in the documentation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from telemetry_anomdet.feature_extraction.features import make_feature_table
from telemetry_anomdet.ingest import load_smap
from telemetry_anomdet.models.deep.codegen import write_c
from telemetry_anomdet.models.deep.distill import KANGDNNumpy, extract_kan_gdn
from telemetry_anomdet.models.deep.kan_gdn import KANGDN
from telemetry_anomdet.preprocessing import pipeline

DATA_DIR = Path(os.environ.get("TAD_SMAP_DIR", "")).expanduser()
CHANNEL = os.environ.get("TAD_SMAP_CHANNEL", "D-2")
DTYPE = os.environ.get("TAD_EXPORT_DTYPE", "float")
N_VECTORS = int(os.environ.get("TAD_EXPORT_VECTORS", "8"))
EMBED_DIM = int(os.environ.get("TAD_KANGDN_EMBED_DIM", "16"))
GRID_SIZE = int(os.environ.get("TAD_KANGDN_GRID_SIZE", "5"))
EPOCHS = int(os.environ.get("TAD_GDN_EPOCHS", "30"))
WINDOW_SIZE = int(os.environ.get("TAD_GDN_WINDOW", "30"))
STEP = 10


def load_windows() -> np.ndarray:
    """Windows to train on: a real SMAP channel when available, else synthetic."""
    if DATA_DIR and (DATA_DIR / "train").exists():
        frame = pipeline(
            load_smap(DATA_DIR, [CHANNEL], split="train", dims="all").to_pandas(),
            resample_rule=None,
        )
        windows = make_feature_table(frame, window_size=WINDOW_SIZE, step=STEP)
        print(f"training on SMAP channel {CHANNEL}: {windows.shape}")
        return windows

    rng = np.random.default_rng(0)
    windows = rng.normal(size=(200, WINDOW_SIZE, 25))
    print(f"TAD_SMAP_DIR unset; training on synthetic data: {windows.shape}")
    return windows


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <output-directory>")
    out_dir = Path(sys.argv[1])

    windows = load_windows()
    detector = KANGDN(
        embed_dim=EMBED_DIM,
        grid_size=GRID_SIZE,
        topk=10,
        epochs=EPOCHS,
        random_state=0,
    )
    detector.fit(windows)

    spec = extract_kan_gdn(detector)
    vectors = windows[:N_VECTORS]
    written = write_c(spec, out_dir, dtype=DTYPE, test_windows=vectors)

    n_nodes = spec["net"]["n_nodes"]
    n_params = sum(
        int(np.asarray(v).size)
        for layer in (spec["net"]["activation"], spec["net"]["out"])
        for k, v in layer.items()
        if isinstance(v, np.ndarray)
    )
    width = 4 if DTYPE == "float" else 8

    print(f"\nwrote {len(written)} files to {out_dir}:")
    for path in written:
        print(f"  {path.name:<22} {path.stat().st_size / 1024:8.1f} KB")
    print(
        f"\nmodel: {n_nodes} channels, embed_dim={EMBED_DIM}, window={WINDOW_SIZE}, dtype={DTYPE}"
    )
    print(f"KAN coefficients: {n_params:,} ({n_params * width / 1024:.1f} KB)")
    print(f"threshold: {spec['threshold']:.6f}")

    # Confirm the evaluator the vectors were taken from still agrees, so a target
    # mismatch can only come from the port.
    check = KANGDNNumpy(spec).decision_function(vectors)
    print(f"reference scores: {np.array2string(check, precision=4)}")


if __name__ == "__main__":
    main()
