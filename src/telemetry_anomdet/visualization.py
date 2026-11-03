# src/telemetry_anomdet/visualization.py

"""
Figures for inspecting a fitted detector.

These are deliberately not general time-series plots; matplotlib already does
those better. Each function here shows something only this toolkit can show:
what the graph detector learned, and which channels drove a score.

matplotlib is an optional dependency (the ``viz`` extra) and is imported inside
each function, so importing this module costs nothing when it is absent.

Every function draws onto an ``Axes`` and returns it, so figures compose into
larger layouts and nothing is written to disk or shown on your behalf.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["plot_sensor_graph"]

# Above this many channels a node-link diagram becomes a hairball: every node
# has topk neighbours and the chords all cross the interior. A matrix shows the
# same information at any size, and unlike a hairball it looks visibly different
# when there is no structure to find.
_MATRIX_THRESHOLD = 12

_DEGENERATE_COLOR = "#c0392b"


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Plotting requires matplotlib. Install the viz extra: "
            'pip install "telemetry-anomdet[viz]"'
        ) from exc
    return plt


def _circle_layout(n: int) -> np.ndarray:
    """
    Node positions evenly spaced on a circle, starting at twelve o'clock.

    A fixed layout rather than a force-directed one, because the point is to
    compare graphs across runs and datasets. A layout that moves with the data
    makes two graphs look different when only the edges changed.
    """
    angles = np.pi / 2 - 2 * np.pi * np.arange(n) / max(n, 1)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _resolve_deviation_norm(deviations: np.ndarray, log_scale: bool):
    """Colour normalisation for deviations, and the values to draw with it."""
    if not log_scale:
        return None, deviations
    from matplotlib.colors import LogNorm

    positive = deviations[deviations > 0]
    if not positive.size:
        return None, deviations
    vmin = float(positive.min())
    norm = LogNorm(vmin=vmin, vmax=max(float(deviations.max()), vmin * 10))
    return norm, np.clip(deviations, vmin, None)


def _dominant_channel(deviations: np.ndarray, scoring: Sequence[int]) -> int:
    """
    The channel whose deviation set the score for this window.

    Restricted to the scoring channels so the answer matches
    ``detector.dominant_channels``; a plain argmax could name a context-only
    channel that never contributed.
    """
    scoring = list(scoring)
    return int(scoring[int(np.argmax(deviations[scoring]))])


def _draw_node_link(
    ax, plt, graph, names, deviations, cmap, log_scale, degenerate, dominant, legend
):
    adjacency = graph["adjacency"]
    similarity = graph["similarity"]
    n_nodes = adjacency.shape[0]
    pos = _circle_layout(n_nodes)

    # Edge shading spans only the similarities actually used, so the contrast
    # reflects preference among selected neighbours rather than the full range.
    used = similarity[adjacency]
    lo, hi = (float(used.min()), float(used.max())) if used.size else (0.0, 1.0)
    span = max(hi - lo, 1e-9)

    # adjacency[i, j] means j is a neighbour of i, so i is forecast *from* j.
    # The arrow is drawn j -> i, pointing where the influence lands, which is
    # the convention everywhere else that draws directed relationships. Two
    # channels that pick each other become one double-headed line rather than
    # a pair of arcs, halving the ink for what is a single mutual relationship.
    drawn: set[tuple[int, int]] = set()
    for target in range(n_nodes):
        for source in np.flatnonzero(adjacency[target]):
            source = int(source)
            mutual = bool(adjacency[source, target])
            key = (min(source, target), max(source, target))
            if mutual and key in drawn:
                continue
            drawn.add(key)

            weight = (similarity[target, source] - lo) / span
            # When explaining one window, the edges into the channel that set
            # the score are what matter; the rest recede rather than disappear.
            touches_dominant = dominant in (source, target)
            emphasis = 1.0 if dominant is None or touches_dominant else 0.25
            ax.annotate(
                "",
                xy=pos[target],
                xytext=pos[source],
                arrowprops=dict(
                    arrowstyle="<|-|>" if mutual else "-|>",
                    color=plt.get_cmap("Greys")(0.35 + 0.5 * weight),
                    alpha=emphasis,
                    linewidth=0.7 + 1.6 * weight,
                    shrinkA=16,
                    shrinkB=16,
                    connectionstyle="arc3,rad=0" if mutual else "arc3,rad=0.12",
                ),
            )

    scoring = graph["scoring"]
    # Degenerate channels get a red rim: they are the ones whose deviations are
    # unbounded, so seeing them beside the scores explains an implausible value.
    edge_colors = [_DEGENERATE_COLOR if i in degenerate else "white" for i in range(n_nodes)]
    line_widths = [2.2 if i in degenerate else 1.0 for i in range(n_nodes)]

    if deviations is None:
        colors = ["#d95f02" if i in scoring else "#cccccc" for i in range(n_nodes)]
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            s=620,
            c=colors,
            zorder=3,
            edgecolors=edge_colors,
            linewidths=line_widths,
        )
    else:
        norm, values = _resolve_deviation_norm(deviations, log_scale)
        scatter = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            s=620,
            c=values,
            cmap=cmap,
            norm=norm,
            zorder=3,
            edgecolors=edge_colors,
            linewidths=line_widths,
        )
        cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label("deviation (training IQRs)" + (", log scale" if norm else ""))

    if dominant is not None:
        # Snug against the marker rather than a halo around it, and drawn
        # above the node so a tight radius is not swallowed by the fill.
        ax.scatter(
            pos[dominant, 0],
            pos[dominant, 1],
            s=880,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.8,
            zorder=4,
        )

    # Labels sit outside the ring rather than inside the markers: real channel
    # names are long enough to overflow a node, and putting them outside also
    # keeps them legible over the deviation fill.
    for i, name in enumerate(names):
        x, y = pos[i]
        if i in degenerate:
            color = _DEGENERATE_COLOR
        elif deviations is not None or i in scoring:
            color = "#222222"
        else:
            color = "#999999"
        ax.annotate(
            name,
            (x * 1.16, y * 1.16),
            ha="left" if x > 0.05 else ("right" if x < -0.05 else "center"),
            va="bottom" if y > 0.05 else ("top" if y < -0.05 else "center"),
            fontsize=8,
            fontweight="bold" if i == dominant else "normal",
            zorder=4,
            color=color,
        )

    if legend:
        # A text key rather than proxy handles: matplotlib's legend markers
        # cannot draw an arrowhead, and a stray ">" glyph explains less than the
        # sentence does. The matrix view states the same on its axes, which is
        # exactly the clarity an unlabelled arrow lacks.
        lines = [
            "→  points to the channel being forecast",
            "↔  the two picked each other",
            "darker and thicker = stronger similarity",
        ]
        if degenerate:
            lines.append("red rim = no spread in training, deviation unbounded")
        if dominant is not None:
            lines.append("black ring = set this window's score")
        ax.text(
            0.0,
            1.0,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            color="#555555",
            linespacing=1.5,
        )

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def _draw_matrix(ax, plt, graph, names, deviations, cmap, log_scale, degenerate, dominant, legend):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    adjacency = graph["adjacency"]
    n_nodes = adjacency.shape[0]
    # The diagonal is -inf so self is never selected; blank it out for drawing.
    similarity = np.where(np.isfinite(graph["similarity"]), graph["similarity"], np.nan)
    limit = float(np.nanmax(np.abs(similarity))) if np.isfinite(similarity).any() else 1.0

    image = ax.imshow(similarity, cmap="RdBu_r", vmin=-limit, vmax=limit)
    rows, cols = np.nonzero(adjacency)
    ax.scatter(
        cols, rows, s=16, marker="s", facecolors="none", edgecolors="#111111", linewidths=0.9
    )

    ax.set_xticks(range(n_nodes))
    ax.set_yticks(range(n_nodes))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    for i in degenerate:
        ax.get_xticklabels()[i].set_color(_DEGENERATE_COLOR)
        ax.get_yticklabels()[i].set_color(_DEGENERATE_COLOR)

    if dominant is not None:
        ax.axhline(dominant, color="#111111", linewidth=1.1, alpha=0.7)
        ax.axvline(dominant, color="#111111", linewidth=1.1, alpha=0.7)
        ax.get_yticklabels()[dominant].set_fontweight("bold")
        ax.get_xticklabels()[dominant].set_fontweight("bold")

    ax.set_xlabel("neighbour")
    ax.set_ylabel("forecast target")

    divider = make_axes_locatable(ax)

    if deviations is None:
        cax = divider.append_axes("right", size="4%", pad=0.15)
        ax.figure.colorbar(image, cax=cax, label="embedding cosine similarity")
        return ax

    # With deviations there are two scales to place. Stacking both colorbars on
    # the right puts each one's label over the next axes, so the similarity
    # scale goes underneath and the right side is left to the deviations.
    #
    # The marginal strip keeps the matrix itself about structure while still
    # showing which channels deviated, so one figure answers both questions.
    norm, values = _resolve_deviation_norm(deviations, log_scale)
    strip = divider.append_axes("right", size="6%", pad=0.12)
    img = strip.imshow(values.reshape(-1, 1), cmap=cmap, norm=norm, aspect="auto")
    strip.set_xticks([])
    strip.set_yticks([])
    strip.set_title("dev.", fontsize=7)

    dev_cax = divider.append_axes("right", size="4%", pad=0.12)
    ax.figure.colorbar(
        img,
        cax=dev_cax,
        label="deviation (training IQRs)" + (", log scale" if norm else ""),
    )

    sim_cax = divider.append_axes("bottom", size="3%", pad=0.9)
    ax.figure.colorbar(
        image, cax=sim_cax, orientation="horizontal", label="embedding cosine similarity"
    )
    return ax


def plot_sensor_graph(
    detector,
    *,
    channel_names: Sequence[str] | None = None,
    deviations: np.ndarray | None = None,
    ax=None,
    cmap: str = "viridis",
    log_scale: bool = False,
    style: str = "auto",
    legend: bool = True,
    title: str | None = "auto",
):
    """
    Draw the relational graph a GDN or KANGDN learned over its channels.

    Each channel is forecast from a handful of neighbours chosen by the cosine
    similarity of learned embeddings, so the figure is what the detector
    believes predicts what. Reading it is the most direct check on whether the
    graph found real structure, which metrics show only indirectly.

    Two encodings are available. The node-link view draws channels on a circle
    with an arrow to each neighbour, which is legible while there are few
    channels. Beyond a dozen every chord crosses the interior and the picture
    becomes a hairball, so the matrix view draws the full similarity matrix with
    the selected neighbours marked. The matrix has a second advantage: a graph
    with real structure shows blocks, and one without shows noise, a difference
    a hairball cannot convey.

    Channels flagged by ``detector.degenerate_channels_()`` are marked in red.
    Those had no measurable spread in training, so their deviations are
    unbounded, and seeing them beside a score explains an implausible value.

    Parameters
    ----------
    detector : GDN or KANGDN
        A fitted detector.
    channel_names : sequence of str, optional
        Labels for the channels. Defaults to positional indices.
    deviations : np.ndarray, optional
        Per-channel deviations for one window, shape ``(n_channels,)``, as
        returned by a row of ``detector.channel_deviations(X)``. When given, the
        figure explains a single alarm: the channel that set the score is
        outlined and its edges emphasised, and the rest recede.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when omitted.
    cmap : str, default="viridis"
        Colormap for deviations.
    log_scale : bool, default=False
        Colour deviations on a log scale. A channel that barely moved while
        training divides by a spread near zero and can outrun the rest by many
        orders of magnitude, flattening every other channel on a linear
        colormap. Zeros are clipped to the smallest positive value present.
    style : {"auto", "graph", "matrix"}, default="auto"
        Which encoding to draw. ``"auto"`` picks node-link up to twelve
        channels and the matrix beyond.
    legend : bool, default=True
        Draw a key for the node-link view. The matrix view states the same
        information on its axes and ignores this. Turn it off when composing
        several panels that share one caption.
    title : str or None, default="auto"
        ``"auto"`` describes the graph and names the channel that set the
        score. Pass a string to replace it, or None for no title, which suits a
        figure whose caption carries the description.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> ax = plot_sensor_graph(detector, channel_names=names)   # doctest: +SKIP
    >>> scores = detector.decision_function(X)                  # doctest: +SKIP
    >>> worst = scores.argmax()                                 # doctest: +SKIP
    >>> ax = plot_sensor_graph(                                 # doctest: +SKIP
    ...     detector,
    ...     deviations=detector.channel_deviations(X)[worst],
    ...     log_scale=True,
    ... )
    """
    plt = _require_matplotlib()

    if style not in {"auto", "graph", "matrix"}:
        raise ValueError(f"style must be 'auto', 'graph' or 'matrix', got {style!r}")

    graph = detector.learned_graph()
    graph["scoring"] = set(detector.scoring_channels_)
    n_nodes = graph["adjacency"].shape[0]

    if channel_names is None:
        channel_names = [str(i) for i in range(n_nodes)]
    elif len(channel_names) != n_nodes:
        raise ValueError(
            f"channel_names has {len(channel_names)} entries but the detector "
            f"was fitted on {n_nodes} channels."
        )

    dominant = None
    if deviations is not None:
        deviations = np.asarray(deviations, dtype=float).ravel()
        if deviations.shape[0] != n_nodes:
            raise ValueError(
                f"deviations has {deviations.shape[0]} entries but the detector "
                f"was fitted on {n_nodes} channels."
            )
        dominant = _dominant_channel(deviations, detector.scoring_channels_)

    degenerate = set(np.asarray(detector.degenerate_channels_()).tolist())

    resolved = style
    if style == "auto":
        resolved = "graph" if n_nodes <= _MATRIX_THRESHOLD else "matrix"

    if ax is None:
        figsize = (6.5, 6.5) if resolved == "graph" else (max(6.0, 0.34 * n_nodes + 3), 6.5)
        _, ax = plt.subplots(figsize=figsize)

    draw = _draw_node_link if resolved == "graph" else _draw_matrix
    draw(ax, plt, graph, channel_names, deviations, cmap, log_scale, degenerate, dominant, legend)

    if title == "auto":
        neighbours = min(detector.topk, n_nodes - 1)
        title = f"Learned sensor graph: {n_nodes} channels, top-{neighbours} neighbours"
        if dominant is not None:
            title += f"\nscore set by {channel_names[dominant]}"
    if title:
        ax.set_title(title)
    return ax
