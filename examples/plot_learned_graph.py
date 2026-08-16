"""
Draw the relational graph a KANGDN learns, on data where the answer is known.

Run it to see what the detector believes predicts what:

    uv run --extra deep --extra viz python examples/plot_learned_graph.py

Figures are written to build/figures/ (gitignored). Pass --show to open them
in a window instead, and an output directory as the first positional argument
to write elsewhere.

Two cases are drawn, because a figure you cannot calibrate is a figure you
cannot trust:

*Coupled*, synthetic, six channels with a structure we choose in advance. Two
pairs are genuinely related and two channels are independent noise. The pairs
should appear as the thick dark reciprocal edges. This is the check that the
figure can show structure at all.

*SMAP*, real telemetry, twenty-five channels of which one is a sensor and the
rest are command flags. There is nothing for a graph to relate, so the edges
should look arbitrary. This is the check that the figure does not invent
structure that is not there, and it is why the benchmark improves when the
graph is removed on this dataset.

Set TAD_SMAP_DIR to a SMAP directory to include the second figure; without it
only the synthetic case is drawn.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from telemetry_anomdet.models.deep.kan_gdn import KANGDN
from telemetry_anomdet.visualization import plot_sensor_graph

WINDOW = 20


def _windowify(raw: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Overlapping windows of shape (n_windows, window, n_channels)."""
    return np.stack([raw[i : i + window] for i in range(0, len(raw) - window, stride)])


def coupled_example(seed: int = 0):
    """Six channels: (A, B) and (C, D) coupled, two independent noise channels."""
    rng = np.random.default_rng(seed)
    n = 4000
    t = np.arange(n)

    a = np.sin(t / 50) + 0.05 * rng.normal(size=n)
    b = 0.9 * a + 0.05 * rng.normal(size=n)
    c = np.sin(t / 17 + 1.0) + 0.05 * rng.normal(size=n)
    d = -0.8 * c + 0.05 * rng.normal(size=n)
    noise1 = rng.normal(size=n)
    noise2 = rng.normal(size=n)

    raw = np.column_stack([a, b, c, d, noise1, noise2])
    names = ["A", "B = f(A)", "C", "D = f(C)", "noise 1", "noise 2"]

    X = _windowify(raw, WINDOW, stride=4)
    det = KANGDN(embed_dim=16, topk=2, epochs=25, batch_size=64, random_state=0)
    det.fit(X)
    return det, X, names


def smap_example(smap_dir: Path, channel: str = "A-1"):
    """One real SMAP channel: one telemetry dimension plus 24 command flags."""
    path = smap_dir / "train" / f"{channel}.npy"
    if not path.exists():  # the loader also accepts the nested layout
        path = smap_dir / "data" / "train" / f"{channel}.npy"
    if not path.exists():
        raise FileNotFoundError(f"No SMAP channel at {path}")

    raw = np.load(path)
    names = ["telemetry"] + [f"cmd {i}" for i in range(1, raw.shape[1])]

    X = _windowify(raw, 30, stride=5)
    det = KANGDN(embed_dim=16, topk=5, epochs=6, batch_size=64, random_state=0)
    det.fit(X)
    return det, X, names


def describe(det, X, names, label: str) -> None:
    """Print what the figure shows, so the run is readable without the image."""
    graph = det.learned_graph()
    similarity = graph["similarity"][graph["adjacency"]]
    offdiag = graph["similarity"].copy()
    np.fill_diagonal(offdiag, np.nan)

    print(f"\n{label}: {len(names)} channels, top-{min(det.topk, len(names) - 1)} neighbours")
    print(
        f"  selected-neighbour similarity  min {similarity.min():+.3f}  max {similarity.max():+.3f}"
    )
    print(
        f"  all-pairs similarity           mean {np.nanmean(offdiag):+.3f}  "
        f"std {np.nanstd(offdiag):.3f}"
    )
    if len(names) <= 8:
        for i, name in enumerate(names):
            neighbours = ", ".join(names[j] for j in np.flatnonzero(graph["adjacency"][i]))
            print(f"    {name:<10} <- {neighbours}")

    worst = int(det.decision_function(X).argmax())
    print(f"  most anomalous window {worst}, driven by {names[det.dominant_channels(X)[worst]]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", default="build/figures")
    parser.add_argument("--show", action="store_true", help="open windows instead of saving")
    args = parser.parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cases = [("coupled", *coupled_example())]

    smap_dir = os.environ.get("TAD_SMAP_DIR")
    if smap_dir:
        try:
            cases.append(("smap", *smap_example(Path(smap_dir))))
        except FileNotFoundError as exc:
            print(f"Skipping SMAP figure: {exc}")
    else:
        print("Set TAD_SMAP_DIR to also draw the SMAP figure.")

    for label, det, X, names in cases:
        describe(det, X, names, label)

        # Titles and the key are left off: what each figure shows is printed
        # above and carried by the filename, and the images are meant to sit
        # under a caption. Pass title="auto" or legend=True to get them back.
        ax = plot_sensor_graph(det, channel_names=names, legend=False, title=None)

        deviations = det.channel_deviations(X)
        worst = int(det.decision_function(X).argmax())
        # Log scale, because a channel that barely moved while training divides
        # by a spread near zero and can outrun the rest by many orders of
        # magnitude, flattening every other node on a linear colormap.
        ax2 = plot_sensor_graph(
            det,
            channel_names=names,
            deviations=deviations[worst],
            log_scale=True,
            legend=False,
            title=None,
        )

        if args.show:
            continue
        for suffix, axes in (("graph", ax), ("alarm", ax2)):
            path = outdir / f"{label}_{suffix}.png"
            axes.figure.savefig(path, dpi=220, bbox_inches="tight")
            print(f"  wrote {path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
