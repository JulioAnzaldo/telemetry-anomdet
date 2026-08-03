import numpy as np
import pytest

# GDN needs the optional deep extra (torch). Skip the whole module if absent.
pytest.importorskip("torch")

from telemetry_anomdet.models.deep.gdn import GDN  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_series(rng, n_windows, window_size=10, n_features=4, scale=1.0, shift=0.0):
    """Smooth per-window sequences so forecasting has signal to learn."""
    base = rng.normal(shift, scale, size=(n_windows, 1, n_features))
    noise = rng.normal(0, 0.05, size=(n_windows, window_size, n_features))
    ramp = np.linspace(0, 1, window_size)[None, :, None]
    return base + ramp * base * 0.1 + noise


def fast_gdn(**kw):
    kw.setdefault("epochs", 3)
    kw.setdefault("embed_dim", 16)
    kw.setdefault("topk", 3)
    kw.setdefault("random_state", 0)
    return GDN(**kw)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def test_gdn_basic_fit_and_scores():
    rng = np.random.default_rng(0)
    X_train = make_series(rng, 120)
    X_test = make_series(rng, 20)

    det = fast_gdn().fit(X_train)
    scores = det.decision_function(X_test)

    assert scores.shape == (20,)
    assert scores.dtype.kind == "f"
    assert np.isfinite(scores).all()


# ---------------------------------------------------------------------------
# Post-fit attributes
# ---------------------------------------------------------------------------


def test_gdn_postfit_attributes():
    rng = np.random.default_rng(1)
    X = make_series(rng, 100)
    det = fast_gdn(percentile=95.0).fit(X)

    assert det.decision_scores_ is not None
    assert det.decision_scores_.shape == (100,)
    assert isinstance(det.threshold_, float)
    assert set(det.labels_).issubset({0, 1})
    assert det.n_nodes_ == 4
    assert det.window_ == 9  # window_size - 1


# ---------------------------------------------------------------------------
# predict returns binary labels
# ---------------------------------------------------------------------------


def test_gdn_predict_returns_binary():
    rng = np.random.default_rng(2)
    X = make_series(rng, 80)
    det = fast_gdn().fit(X)
    preds = det.predict(X)
    assert preds.shape == (80,)
    assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# Anomalous windows score higher
# ---------------------------------------------------------------------------


def test_gdn_anomalous_scores_higher():
    rng = np.random.default_rng(3)
    X_train = make_series(rng, 200, scale=1.0)
    X_normal = make_series(rng, 40, scale=1.0)
    X_anom = make_series(rng, 40, scale=1.0, shift=8.0)  # off-distribution level

    det = fast_gdn(epochs=8).fit(X_train)

    assert det.decision_function(X_anom).mean() > det.decision_function(X_normal).mean()


# ---------------------------------------------------------------------------
# Determinism with a fixed seed
# ---------------------------------------------------------------------------


def test_gdn_reproducible_with_seed():
    rng = np.random.default_rng(4)
    X = make_series(rng, 80)

    a = fast_gdn(random_state=123).fit(X).decision_scores_
    b = fast_gdn(random_state=123).fit(X).decision_scores_
    np.testing.assert_allclose(a, b)


# ---------------------------------------------------------------------------
# is_anomaly overrides (inherited from BaseDetector)
# ---------------------------------------------------------------------------


def test_gdn_is_anomaly_threshold_override():
    rng = np.random.default_rng(5)
    X = make_series(rng, 80)
    det = fast_gdn().fit(X)

    assert det.is_anomaly(X, threshold=-999.0).all()
    assert not det.is_anomaly(X, threshold=1e9).any()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_gdn_rejects_2d_input():
    rng = np.random.default_rng(6)
    det = fast_gdn().fit(make_series(rng, 50))
    with pytest.raises(ValueError, match="3D"):
        det.decision_function(np.ones((50, 4)))


def test_gdn_rejects_window_size_1():
    det = fast_gdn()
    with pytest.raises(ValueError, match="window_size >= 2"):
        det.fit(np.ones((20, 1, 4)))


def test_gdn_rejects_feature_mismatch():
    rng = np.random.default_rng(7)
    det = fast_gdn().fit(make_series(rng, 50, n_features=4))
    with pytest.raises(ValueError, match="fitted on 4"):
        det.decision_function(make_series(rng, 10, n_features=5))


def test_gdn_requires_fit_before_decision():
    with pytest.raises(RuntimeError, match="not fitted"):
        GDN().decision_function(np.ones((10, 5, 4)))


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_gdn_repr():
    det = fast_gdn()
    assert "fitted=False" in repr(det)
    det.fit(make_series(np.random.default_rng(0), 40))
    assert "fitted=True" in repr(det)


# ---------------------------------------------------------------------------
# Internal scaler
# ---------------------------------------------------------------------------


def test_gdn_scale_default_fits_scaler():
    rng = np.random.default_rng(10)
    det = fast_gdn().fit(make_series(rng, 60))
    assert det.scale is True
    assert det.scaler is not None
    # Per-channel scaler: one mean/scale per feature (node).
    assert det.scaler.mean_.shape == (4,)


def test_gdn_scale_false_skips_scaler():
    rng = np.random.default_rng(11)
    det = fast_gdn(scale=False).fit(make_series(rng, 60))
    assert det.scaler is None
    scores = det.decision_function(make_series(rng, 15))
    assert scores.shape == (15,)
    assert np.isfinite(scores).all()


def test_gdn_scale_handles_constant_channel():
    # One channel is a constant (e.g. an inactive command one-hot): zero
    # variance must not produce NaNs/Infs via divide-by-zero in the scaler.
    rng = np.random.default_rng(12)
    X = make_series(rng, 80, n_features=4)
    X[:, :, 3] = 1.0  # channel 3 is constant across every window and timestep

    det = fast_gdn().fit(X)
    scores = det.decision_function(X)
    assert np.isfinite(scores).all()


def test_gdn_scale_makes_detector_scale_invariant():
    # With scaling on, multiplying one channel by a large constant should not
    # blow up the scores (the whole point of standardising inputs).
    rng = np.random.default_rng(13)
    X = make_series(rng, 120, n_features=4)

    base = fast_gdn(random_state=7).fit(X).decision_scores_

    X_scaled = X.copy()
    X_scaled[:, :, 0] *= 1000.0
    blown = fast_gdn(random_state=7).fit(X_scaled).decision_scores_

    assert np.isfinite(blown).all()
    # Scores stay in the same ballpark rather than being dominated by channel 0.
    assert blown.max() < base.max() * 10


def test_gdn_scale_reproducible_with_seed():
    rng = np.random.default_rng(14)
    X = make_series(rng, 80)
    a = fast_gdn(random_state=99).fit(X).decision_scores_
    b = fast_gdn(random_state=99).fit(X).decision_scores_
    np.testing.assert_allclose(a, b)


def test_gdn_scale_shows_in_repr():
    assert "scale=False" in repr(fast_gdn(scale=False))
    assert "scale=True" in repr(fast_gdn(scale=True))
