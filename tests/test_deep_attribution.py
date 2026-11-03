import warnings

import numpy as np
import pytest

pytest.importorskip("torch")

from telemetry_anomdet.models.deep.kan_gdn import KANGDN  # noqa: E402

WINDOW_SIZE = 6
N_FEATURES = 5


def _fitted(*, score_channels=None, seed=0, n_windows=40):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_windows, WINDOW_SIZE, N_FEATURES))
    det = KANGDN(
        embed_dim=8,
        topk=3,
        epochs=3,
        batch_size=16,
        random_state=0,
        score_channels=score_channels,
    )
    det.fit(X)
    return det, X


@pytest.fixture
def fitted_pair():
    """A freshly fitted detector, so tests may mutate its post-fit state."""
    return _fitted()


def test_shape_is_one_column_per_channel():
    det, X = _fitted()
    dev = det.channel_deviations(X)
    assert dev.shape == (X.shape[0], N_FEATURES)
    assert np.isfinite(dev).all()
    assert (dev >= 0).all()


def test_all_channels_returned_even_when_scoring_is_restricted():
    """
    Context-only channels are still reported. They cannot raise an alarm but
    they are often what explains one.
    """
    det, X = _fitted(score_channels=[0, 1])
    dev = det.channel_deviations(X)
    assert dev.shape[1] == N_FEATURES
    assert det.scoring_channels_ == [0, 1]


def test_scoring_channels_defaults_to_every_channel():
    det, _ = _fitted()
    assert det.scoring_channels_ == list(range(N_FEATURES))


def test_max_over_scoring_channels_reproduces_the_score():
    """
    The score is the maximum deviation over the scoring channels. If this drifts
    the attribution no longer explains the number it claims to explain.
    """
    for channels in (None, [0, 2, 4]):
        det, X = _fitted(score_channels=channels)
        dev = det.channel_deviations(X)
        expected = dev[:, det.scoring_channels_].max(axis=1)
        np.testing.assert_allclose(det.decision_function(X), expected, rtol=1e-10)


def test_dominant_channel_carries_the_score():
    det, X = _fitted()
    dev = det.channel_deviations(X)
    dominant = det.dominant_channels(X)
    picked = dev[np.arange(len(dominant)), dominant]
    np.testing.assert_allclose(picked, det.decision_function(X), rtol=1e-10)


def test_dominant_channel_never_names_a_context_only_channel():
    """
    Regression guard. argmax over the full matrix could name a channel excluded
    from scoring, which would attribute an alarm to something that did not
    contribute to it.
    """
    scoring = [0, 1]
    det, X = _fitted(score_channels=scoring)
    dominant = det.dominant_channels(X)
    assert set(np.unique(dominant)).issubset(set(scoring))


def test_deviations_are_deterministic():
    det, X = _fitted()
    np.testing.assert_array_equal(det.channel_deviations(X), det.channel_deviations(X))


def test_geometry_is_validated():
    det, _ = _fitted()
    wrong_features = np.zeros((4, WINDOW_SIZE, N_FEATURES + 1))
    with pytest.raises(ValueError, match="features but GDN was fitted"):
        det.channel_deviations(wrong_features)

    wrong_window = np.zeros((4, WINDOW_SIZE + 3, N_FEATURES))
    with pytest.raises(ValueError, match="window_size"):
        det.channel_deviations(wrong_window)


def test_learned_graph_shape_and_degree():
    det, _ = _fitted()
    g = det.learned_graph()
    assert g["adjacency"].shape == (N_FEATURES, N_FEATURES)
    assert g["embeddings"].shape[0] == N_FEATURES
    # topk neighbours per node, clamped to n_nodes - 1, and never itself.
    assert (g["adjacency"].sum(axis=1) == min(det.topk, N_FEATURES - 1)).all()
    assert not g["adjacency"].diagonal().any()


def test_learned_graph_matches_the_deployed_artifact():
    """
    The graph you inspect must be the graph that flies. The distilled adjacency
    adds self-loops, because attention there runs over topk + 1 terms, so the
    comparison clears the diagonal.
    """
    from telemetry_anomdet.models.deep.distill import extract_kan_gdn

    det, _ = _fitted()
    distilled = extract_kan_gdn(det)["net"]["adj"].copy()
    np.fill_diagonal(distilled, False)
    np.testing.assert_array_equal(det.learned_graph()["adjacency"], distilled)


# ---------------------------------------------------------------------------
# Channels with no training spread
# ---------------------------------------------------------------------------


def _flatten_spread(det, channel: int):
    """
    Reproduce a channel whose training error had no spread, floor included.

    The state is set directly rather than trained into, because reproducing it
    needs a channel whose graph neighbours are also constant. Making a single
    input channel constant is not enough: attention from the moving channels
    keeps the forecast wobbling and leaves an IQR around 1e-2. It does occur on
    real data, though. Fitting on SMAP channel A-1 leaves 3 of 25 channels with
    an IQR of exactly zero.
    """
    det._err_iqr_ = det._err_iqr_.copy()
    det._err_iqr_[channel] = 0.0
    det._err_iqr_ = det._apply_spread_floor(det._err_iqr_)
    return det


def test_zero_spread_channel_is_reported_as_degenerate(fitted_pair):
    det, _ = fitted_pair
    _flatten_spread(det, 2)
    assert 2 in det.degenerate_channels_()


def test_warning_names_the_offending_channel(fitted_pair):
    det, _ = fitted_pair
    _flatten_spread(det, 2)
    with pytest.warns(RuntimeWarning, match=r"Channels \[2\].*no measurable spread"):
        det._warn_degenerate_spread()


def test_no_warning_when_the_channel_cannot_raise_an_alarm():
    """
    Excluding it via score_channels is the documented workaround, so a detector
    already configured that way should stay quiet.
    """
    det, _ = _fitted(score_channels=[0, 1])
    _flatten_spread(det, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        det._warn_degenerate_spread()
    # Still reported as degenerate; it just cannot affect a score.
    assert 2 in det.degenerate_channels_()


def test_well_behaved_data_does_not_warn():
    det = KANGDN(embed_dim=8, topk=3, epochs=3, batch_size=16, random_state=0)
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        det.fit(rng.normal(size=(40, WINDOW_SIZE, N_FEATURES)))
    assert det.degenerate_channels_().size == 0


def test_ordinary_differences_in_predictability_do_not_warn(fitted_pair):
    """
    The guard targets denominators the epsilon dominates, not channels that are
    merely harder to forecast. A channel an order of magnitude below the median
    is normal and must stay quiet, or the warning becomes noise on real
    telemetry where channels differ widely.
    """
    det, _ = fitted_pair
    det._err_iqr_ = det._err_iqr_.copy()
    det._err_iqr_[2] = np.median(det._err_iqr_) / 50
    assert det.degenerate_channels_().size == 0


def test_spread_floor_keeps_a_degenerate_channel_in_scale(fitted_pair):
    """
    Without the floor, dividing by a spread the guard epsilon dominates gave a
    number set by the epsilon rather than the data: 1.7e10 on SMAP, which won
    every window's maximum and produced 245 false alarms. Floored, the channel
    still scores highly when it moves, but within the same order as the other
    channels rather than a million times above them.
    """
    det, X = fitted_pair
    _flatten_spread(det, 2)

    dev = det.channel_deviations(X)
    others = np.delete(dev, 2, axis=1).max()
    assert dev[:, 2].max() < 100 * others


def test_the_floor_scales_with_the_channels_own_error(fitted_pair):
    """
    The floor is a fraction of the channel's own median error, not a constant,
    so it means the same thing on channels carrying different units.
    """
    det, _ = fitted_pair
    _flatten_spread(det, 2)
    expected = det._SPREAD_FLOOR_RATIO * det._err_median_[2]
    assert det._err_iqr_[2] == pytest.approx(expected)


def test_the_floor_leaves_healthy_channels_untouched(fitted_pair):
    """
    Set below the smallest spread a real channel shows, so it only ever reaches
    channels with none and never reshapes one that has some.
    """
    det, _ = fitted_pair
    raw = np.array([0.0, 0.5, 1.0, 2.0, 4.0]) * det._err_median_
    floored = det._apply_spread_floor(raw.copy())
    np.testing.assert_allclose(floored[1:], raw[1:])
    assert det.degenerate_channels_().tolist() == [0]


def test_requires_a_fitted_detector():
    det = KANGDN(embed_dim=8, topk=3, epochs=1)
    with pytest.raises(RuntimeError, match="is not fitted"):
        det.channel_deviations(np.zeros((2, WINDOW_SIZE, N_FEATURES)))
    with pytest.raises(RuntimeError, match="is not fitted"):
        det.learned_graph()
