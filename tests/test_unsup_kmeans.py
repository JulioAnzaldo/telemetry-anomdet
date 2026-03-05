import numpy as np
import pytest
from telemetry_anomdet.models.unsupervised.kmeans import KMeansAnomaly

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def make_X3d(rng, center, n_windows, window_size=5, n_features=2):
    """Build a 3D window tensor where every time step in a window is drawn
    from a Gaussian centered at `center`."""
    X2d = rng.normal(center, 0.3, size=(n_windows, n_features))
    return np.tile(X2d[:, np.newaxis, :], (1, window_size, 1))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_kmeans_basic_fit_and_scores():
    rng = np.random.default_rng(0)
    X_train = np.concatenate([
        make_X3d(rng, [0, 0], 100),
        make_X3d(rng, [3, 3], 100),
    ])                                          # (200, 5, 2)

    X_out   = make_X3d(rng, [8, 8], 10)
    X_test  = np.concatenate([X_train[:20], X_out])  # (30, 5, 2)

    det = KMeansAnomaly(n_clusters=2, scale=True, percentile=95.0).fit(X_train)

    # predict_clusters returns valid cluster indices
    labels = det.predict_clusters(X_test)
    assert labels.shape == (X_test.shape[0],)
    assert set(labels).issubset({0, 1})

    # decision_function returns float scores
    scores = det.decision_function(X_test)
    assert scores.shape == (X_test.shape[0],)
    assert scores.dtype.kind == "f"

    # predict returns binary int labels
    preds = det.predict(X_test)
    assert preds.shape == (X_test.shape[0],)
    assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# Post-fit attributes
# ---------------------------------------------------------------------------

def test_kmeans_postfit_attributes():
    rng = np.random.default_rng(1)
    X = np.concatenate([make_X3d(rng, [0, 0], 80), make_X3d(rng, [5, 5], 80)])

    det = KMeansAnomaly(n_clusters=2).fit(X)

    assert det.decision_scores_ is not None
    assert det.decision_scores_.shape == (X.shape[0],)
    assert det.threshold_ is not None
    assert isinstance(det.threshold_, float)
    assert det.labels_ is not None
    assert set(det.labels_).issubset({0, 1})


# ---------------------------------------------------------------------------
# Anomalous windows score higher than normal windows
# ---------------------------------------------------------------------------

def test_kmeans_anomalous_scores_higher():
    rng = np.random.default_rng(2)
    X_train  = np.concatenate([make_X3d(rng, [0, 0], 100), make_X3d(rng, [3, 3], 100)])
    X_normal = make_X3d(rng, [0, 0], 50)
    X_anom   = make_X3d(rng, [20, 20], 50)   # far outside training distribution

    det = KMeansAnomaly(n_clusters=2, scale=True).fit(X_train)

    assert det.decision_function(X_anom).mean() > det.decision_function(X_normal).mean()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_kmeans_rejects_2d_input():
    det = KMeansAnomaly().fit(np.random.default_rng(0).normal(size=(50, 5, 2)))
    with pytest.raises(ValueError, match="3D"):
        det.decision_function(np.ones((50, 2)))


def test_kmeans_rejects_nan_input():
    X = np.random.default_rng(0).normal(size=(50, 5, 2))
    X[3, 2, 0] = np.nan
    with pytest.raises(ValueError):
        KMeansAnomaly().fit(X)


def test_kmeans_requires_fit_before_predict():
    det = KMeansAnomaly()
    with pytest.raises(RuntimeError, match="not fitted"):
        det.decision_function(np.ones((10, 5, 2)))


# ---------------------------------------------------------------------------
# n_clusters validation
# ---------------------------------------------------------------------------

def test_kmeans_n_clusters_exceeds_windows_raises():
    X = np.random.default_rng(0).normal(size=(5, 5, 2))
    with pytest.raises(ValueError, match="n_clusters"):
        KMeansAnomaly(n_clusters=10).fit(X)


def test_kmeans_n_clusters_zero_raises():
    X = np.random.default_rng(0).normal(size=(50, 5, 2))
    with pytest.raises(ValueError, match="n_clusters"):
        KMeansAnomaly(n_clusters=0).fit(X)


# ---------------------------------------------------------------------------
# scale=False variant
# ---------------------------------------------------------------------------

def test_kmeans_scale_false():
    rng = np.random.default_rng(3)
    X = make_X3d(rng, [0, 0], 80)
    det = KMeansAnomaly(n_clusters=2, scale=False).fit(X)
    assert det.scaler is None
    scores = det.decision_function(X)
    assert scores.shape == (80,)


# ---------------------------------------------------------------------------
# Repr shows fitted state
# ---------------------------------------------------------------------------

def test_kmeans_repr():
    det = KMeansAnomaly(n_clusters=3)
    assert "fitted=False" in repr(det)
    det.fit(np.random.default_rng(0).normal(size=(30, 5, 2)))
    assert "fitted=True" in repr(det)
