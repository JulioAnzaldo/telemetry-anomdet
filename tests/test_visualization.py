import numpy as np
import pytest

pytest.importorskip("torch")
plt = pytest.importorskip("matplotlib.pyplot")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from telemetry_anomdet.models.deep.kan_gdn import KANGDN  # noqa: E402
from telemetry_anomdet.visualization import plot_sensor_graph  # noqa: E402

WINDOW_SIZE = 6
N_FEATURES = 5


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, WINDOW_SIZE, N_FEATURES))
    det = KANGDN(embed_dim=8, topk=2, epochs=3, batch_size=16, random_state=0)
    det.fit(X)
    return det, X


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _expected_lines(adjacency):
    """
    Lines drawn after mutual pairs are merged.

    Two channels that pick each other are one double-headed line, not two arcs,
    so the count is the mutual pairs plus the one-way edges.
    """
    mutual = adjacency & adjacency.T
    one_way = adjacency & ~adjacency.T
    return int(mutual.sum()) // 2 + int(one_way.sum())


def test_returns_axes_and_merges_mutual_pairs(fitted):
    det, _ = fitted
    ax = plot_sensor_graph(det, legend=False)
    assert ax is not None
    adjacency = det.learned_graph()["adjacency"]
    # One annotation per drawn line, plus one label per channel.
    assert len(ax.texts) == _expected_lines(adjacency) + N_FEATURES
    # Merging must actually be happening, or the count above proves nothing.
    assert _expected_lines(adjacency) < int(adjacency.sum())


def test_legend_is_one_text_block_and_can_be_turned_off(fitted):
    det, _ = fitted
    with_legend = plot_sensor_graph(det, legend=True)
    without = plot_sensor_graph(det, legend=False)
    assert len(with_legend.texts) == len(without.texts) + 1
    key = next(t.get_text() for t in with_legend.texts if "forecast" in t.get_text())
    assert "→" in key and "↔" in key


def test_title_can_be_replaced_or_removed(fitted):
    det, _ = fitted
    assert "Learned sensor graph" in plot_sensor_graph(det).get_title()
    assert plot_sensor_graph(det, title=None).get_title() == ""
    assert plot_sensor_graph(det, title="custom").get_title() == "custom"


def test_matrix_ignores_the_legend_flag(fitted):
    """It states the same information on its axes, so a key would be noise."""
    det, _ = fitted
    with_legend = plot_sensor_graph(det, style="matrix", legend=True)
    without = plot_sensor_graph(det, style="matrix", legend=False)
    assert len(with_legend.texts) == len(without.texts)
    assert with_legend.get_ylabel() == "forecast target"


def test_accepts_channel_names(fitted):
    det, _ = fitted
    names = [f"ch_{i}" for i in range(N_FEATURES)]
    ax = plot_sensor_graph(det, channel_names=names)
    drawn = {t.get_text() for t in ax.texts}
    assert set(names).issubset(drawn)


def test_rejects_wrong_length_channel_names(fitted):
    det, _ = fitted
    with pytest.raises(ValueError, match="channel_names has 2 entries"):
        plot_sensor_graph(det, channel_names=["a", "b"])


def test_deviations_add_a_colorbar(fitted):
    det, X = fitted
    ax = plot_sensor_graph(det, deviations=det.channel_deviations(X)[0])
    # The colorbar is a second Axes on the same figure.
    assert len(ax.figure.axes) == 2


def test_rejects_wrong_length_deviations(fitted):
    det, _ = fitted
    with pytest.raises(ValueError, match="deviations has 2 entries"):
        plot_sensor_graph(det, deviations=np.zeros(2))


def test_log_scale_uses_a_log_norm(fitted):
    """
    A single degenerate channel can outrun the rest by orders of magnitude and
    flatten every other node on a linear scale.
    """
    from matplotlib.colors import LogNorm

    det, _ = fitted
    deviations = np.array([1e-3, 1.0, 5.0, 2e4, 0.0])
    ax = plot_sensor_graph(det, deviations=deviations, log_scale=True)
    assert isinstance(ax.collections[0].norm, LogNorm)


def test_log_scale_clips_zeros_rather_than_failing(fitted):
    det, _ = fitted
    ax = plot_sensor_graph(det, deviations=np.zeros(N_FEATURES), log_scale=True)
    assert ax is not None


def test_linear_scale_is_the_default(fitted):
    from matplotlib.colors import LogNorm

    det, X = fitted
    ax = plot_sensor_graph(det, deviations=det.channel_deviations(X)[0])
    assert not isinstance(ax.collections[0].norm, LogNorm)


def test_draws_onto_a_supplied_axes(fitted):
    det, _ = fitted
    fig, ax = plt.subplots()
    returned = plot_sensor_graph(det, ax=ax)
    assert returned is ax


def _fit_wide(n_features, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, WINDOW_SIZE, n_features))
    det = KANGDN(embed_dim=8, topk=2, epochs=2, batch_size=16, random_state=0)
    det.fit(X)
    return det


# ---------------------------------------------------------------------------
# Encoding choice
# ---------------------------------------------------------------------------


def test_auto_uses_node_link_while_channels_are_few(fitted):
    det, _ = fitted
    ax = plot_sensor_graph(det, style="auto")
    # Only the matrix view labels its axes; the node-link view turns them off.
    assert ax.get_xlabel() == ""
    assert not ax.axison


def test_auto_switches_to_matrix_when_channels_are_many():
    """
    Past a dozen channels every chord crosses the interior and the node-link
    view becomes a hairball, which looks identical whether or not the graph
    found structure.
    """
    det = _fit_wide(16)
    ax = plot_sensor_graph(det, style="auto")
    assert ax.get_xlabel() == "neighbour"
    assert len(ax.get_xticks()) == 16


def test_style_can_be_forced_either_way(fitted):
    det, _ = fitted
    assert plot_sensor_graph(det, style="matrix").get_xlabel() == "neighbour"
    assert not plot_sensor_graph(det, style="graph").axison


def test_unknown_style_raises(fitted):
    det, _ = fitted
    with pytest.raises(ValueError, match="style must be"):
        plot_sensor_graph(det, style="spiral")


def test_matrix_marks_the_selected_neighbours(fitted):
    det, _ = fitted
    ax = plot_sensor_graph(det, style="matrix")
    marks = ax.collections[0].get_offsets()
    assert len(marks) == int(det.learned_graph()["adjacency"].sum())


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


def test_degenerate_channels_are_marked_in_both_styles(fitted):
    det, _ = fitted
    det._err_iqr_ = det._err_iqr_.copy()
    det._err_iqr_[3] = 0.0
    # Route through the fit-time path so the floored channel is recorded.
    det._err_iqr_ = det._apply_spread_floor(det._err_iqr_)
    assert 3 in det.degenerate_channels_()

    names = [f"ch{i}" for i in range(N_FEATURES)]
    graph_labels = {
        t.get_text(): t.get_color() for t in plot_sensor_graph(det, channel_names=names).texts
    }
    assert graph_labels["ch3"] != graph_labels["ch0"]

    ax = plot_sensor_graph(det, channel_names=names, style="matrix")
    colors = [t.get_color() for t in ax.get_yticklabels()]
    assert colors[3] != colors[0]


def test_dominant_channel_is_called_out(fitted):
    det, X = fitted
    deviations = det.channel_deviations(X)[0]
    expected = det.dominant_channels(X)[0]
    names = [f"ch{i}" for i in range(N_FEATURES)]

    ax = plot_sensor_graph(det, channel_names=names, deviations=deviations)
    assert f"score set by ch{expected}" in ax.get_title()

    bold = [t.get_text() for t in ax.texts if t.get_fontweight() == "bold"]
    assert bold == [f"ch{expected}"]


def test_dominant_channel_respects_score_channels():
    """
    The figure must name the channel that actually set the score, not whichever
    deviated most overall, or it explains an alarm with a channel that could not
    have caused it.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, WINDOW_SIZE, N_FEATURES))
    det = KANGDN(embed_dim=8, topk=2, epochs=2, batch_size=16, random_state=0, score_channels=[0])
    det.fit(X)

    deviations = np.zeros(N_FEATURES)
    deviations[4] = 99.0  # a context-only channel, excluded from scoring
    ax = plot_sensor_graph(det, deviations=deviations)
    assert "score set by 0" in ax.get_title()


def test_nothing_is_shown_or_written(fitted, tmp_path, monkeypatch):
    """
    Figure functions compose. They must not call show() or write files, or they
    cannot be embedded in a larger layout or a headless report.
    """
    called = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: called.append("show"))
    plot_sensor_graph(fitted[0])
    assert not called
    assert not list(tmp_path.iterdir())
