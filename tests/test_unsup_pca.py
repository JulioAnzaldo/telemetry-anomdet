import numpy as np
import pytest
from telemetry_anomdet.models.unsupervised.pca import PCAAnomaly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_X3d(rng, center, n_windows, window_size=10, n_features=4):
    X2d = rng.normal(center, 0.3, size=(n_windows, n_features))
    return np.tile(X2d[:, np.newaxis, :], (1, window_size, 1))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_pca_basic_fit_and_scores():
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, size=(100, 10, 4))
    X_test  = rng.normal(0, 1, size=(20, 10, 4))

    det = PCAAnomaly().fit(X_train)
    scores = det.decision_function(X_test)

    assert scores.shape == (20,)
    assert scores.dtype.kind == "f"
    assert np.isfinite(scores).all()


# ---------------------------------------------------------------------------
# Post-fit attributes
# ---------------------------------------------------------------------------

def test_pca_postfit_attributes():
    X = np.random.default_rng(1).normal(size=(80, 10, 4))
    det = PCAAnomaly(percentile=95.0).fit(X)

    assert det.decision_scores_ is not None
    assert det.decision_scores_.shape == (80,)
    assert det.threshold_ is not None
    assert isinstance(det.threshold_, float)
    assert det.labels_ is not None
    assert set(det.labels_).issubset({0, 1})


# ---------------------------------------------------------------------------
# predict returns binary labels
# ---------------------------------------------------------------------------

def test_pca_predict_returns_binary():
    X = np.random.default_rng(2).normal(size=(60, 10, 4))
    det = PCAAnomaly().fit(X)
    preds = det.predict(X)
    assert preds.shape == (60,)
    assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# Anomalous windows score higher
# ---------------------------------------------------------------------------

def test_pca_anomalous_scores_higher():
    rng = np.random.default_rng(3)
    X_train  = make_X3d(rng, [0, 0, 0, 0], 200)
    X_normal = make_X3d(rng, [0, 0, 0, 0], 50)
    X_anom   = make_X3d(rng, [15, 15, 15, 15], 50)   # far outside subspace

    det = PCAAnomaly(n_components=2).fit(X_train)

    assert det.decision_function(X_anom).mean() > det.decision_function(X_normal).mean()


# ---------------------------------------------------------------------------
# n_components parameter
# ---------------------------------------------------------------------------

def test_pca_n_components_respected():
    X = np.random.default_rng(4).normal(size=(100, 10, 6))
    det = PCAAnomaly(n_components=2).fit(X)
    assert det.model.n_components_ == 2


# ---------------------------------------------------------------------------
# scale=False variant
# ---------------------------------------------------------------------------

def test_pca_scale_false():
    X = np.random.default_rng(5).normal(size=(60, 10, 4))
    det = PCAAnomaly(scale=False).fit(X)
    assert det.scaler is None
    scores = det.decision_function(X)
    assert scores.shape == (60,)


# ---------------------------------------------------------------------------
# is_anomaly threshold and percentile overrides (inherited)
# ---------------------------------------------------------------------------

def test_pca_is_anomaly_threshold_override():
    X = np.random.default_rng(6).normal(size=(100, 10, 4))
    det = PCAAnomaly().fit(X)

    mask_all  = det.is_anomaly(X, threshold=-999.0)
    mask_none = det.is_anomaly(X, threshold=999.0)
    assert mask_all.all()
    assert not mask_none.any()

def test_pca_is_anomaly_percentile_override():
    X = np.random.default_rng(7).normal(size=(200, 10, 4))
    det = PCAAnomaly().fit(X)
    mask = det.is_anomaly(X, percentile=99.0)
    assert mask.sum() <= 4   # roughly 1% of 200


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_pca_rejects_2d_input():
    det = PCAAnomaly().fit(np.random.default_rng(0).normal(size=(50, 10, 4)))
    with pytest.raises(ValueError, match="3D"):
        det.decision_function(np.ones((50, 4)))

def test_pca_requires_fit_before_predict():
    with pytest.raises(RuntimeError, match="not fitted"):
        PCAAnomaly().decision_function(np.ones((10, 5, 4)))


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

def test_pca_repr():
    det = PCAAnomaly(n_components=3)
    assert "fitted=False" in repr(det)
    det.fit(np.random.default_rng(0).normal(size=(40, 10, 4)))
    assert "fitted=True" in repr(det)
