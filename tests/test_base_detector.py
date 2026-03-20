"""
Tests for BaseDetector shared interface.

Uses a minimal concrete subclass (DummyDetector) that returns the mean
of all values in the flattened window as its anomaly score.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from telemetry_anomdet.models.base import BaseDetector


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing
# ---------------------------------------------------------------------------

class DummyDetector(BaseDetector):
    """Scores each window by its per-element mean. Higher mean = more anomalous."""

    def fit(self, X, y=None):
        X = self._validate_X(X)
        scores = X.mean(axis=(1, 2))
        self._set_post_fit(scores)
        return self

    def decision_function(self, X):
        self._require_fit()
        X = self._validate_X(X)
        return X.mean(axis=(1, 2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normal_X(n=100, w=10, f=4, seed=0):
    return np.random.default_rng(seed).normal(0.0, 1.0, size=(n, w, f))


# ---------------------------------------------------------------------------
# percentile validation
# ---------------------------------------------------------------------------

def test_invalid_percentile_zero():
    with pytest.raises(ValueError, match="percentile"):
        DummyDetector(percentile=0.0)

def test_invalid_percentile_100():
    with pytest.raises(ValueError, match="percentile"):
        DummyDetector(percentile=100.0)

def test_invalid_percentile_negative():
    with pytest.raises(ValueError, match="percentile"):
        DummyDetector(percentile=-5.0)


# ---------------------------------------------------------------------------
# _validate_X
# ---------------------------------------------------------------------------

def test_validate_X_rejects_1d():
    det = DummyDetector().fit(normal_X())
    with pytest.raises(ValueError, match="3D"):
        det.decision_function(np.ones(10))

def test_validate_X_rejects_2d():
    det = DummyDetector().fit(normal_X())
    with pytest.raises(ValueError, match="3D"):
        det.decision_function(np.ones((10, 4)))

def test_validate_X_rejects_nan():
    X = normal_X()
    X[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        DummyDetector().fit(X)

def test_validate_X_rejects_inf():
    X = normal_X()
    X[5, 2, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        DummyDetector().fit(X)

def test_validate_X_rejects_zero_windows():
    with pytest.raises(ValueError, match="0 windows"):
        DummyDetector().fit(np.ones((0, 10, 4)))

def test_validate_X_rejects_zero_features():
    with pytest.raises(ValueError, match="0 features"):
        DummyDetector().fit(np.ones((10, 5, 0)))


# ---------------------------------------------------------------------------
# _require_fit / _get_threshold
# ---------------------------------------------------------------------------

def test_require_fit_raises_before_fit():
    det = DummyDetector()
    with pytest.raises(RuntimeError, match="not fitted"):
        det.decision_function(normal_X(10))

def test_get_threshold_raises_before_fit():
    det = DummyDetector()
    with pytest.raises(RuntimeError):
        det._get_threshold()


# ---------------------------------------------------------------------------
# Post-fit attributes
# ---------------------------------------------------------------------------

def test_postfit_attributes_set():
    X = normal_X(80)
    det = DummyDetector(percentile=95.0).fit(X)

    assert det.decision_scores_ is not None
    assert det.decision_scores_.shape == (80,)
    assert isinstance(det.threshold_, float)
    assert det.labels_ is not None
    assert det.labels_.shape == (80,)
    assert set(det.labels_).issubset({0, 1})

def test_postfit_threshold_matches_percentile():
    X = normal_X(200)
    det = DummyDetector(percentile=90.0).fit(X)
    expected = float(np.percentile(det.decision_scores_, 90.0))
    assert np.isclose(det.threshold_, expected)

def test_postfit_labels_fraction():
    """With percentile=90, at most ~10% of training windows should be labelled 1."""
    X = normal_X(200)
    det = DummyDetector(percentile=90.0).fit(X)
    anomaly_frac = det.labels_.mean()
    assert anomaly_frac <= 0.12   # allow small rounding margin


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_predict_returns_binary_int_array():
    X = normal_X(50)
    det = DummyDetector().fit(X)
    preds = det.predict(X)
    assert preds.shape == (50,)
    assert preds.dtype.kind in ("i", "u")
    assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# is_anomaly
# ---------------------------------------------------------------------------

def test_is_anomaly_default_threshold():
    X = normal_X(100)
    det = DummyDetector(percentile=95.0).fit(X)
    mask = det.is_anomaly(X)
    assert mask.dtype == bool
    assert mask.shape == (100,)

def test_is_anomaly_threshold_override():
    X = normal_X(100)
    det = DummyDetector().fit(X)
    # Force everything to be flagged
    mask_all = det.is_anomaly(X, threshold=-999.0)
    assert mask_all.all()
    # Force nothing to be flagged
    mask_none = det.is_anomaly(X, threshold=999.0)
    assert not mask_none.any()

def test_is_anomaly_percentile_override():
    X = normal_X(200)
    det = DummyDetector().fit(X)
    mask = det.is_anomaly(X, percentile=99.0)
    assert mask.sum() <= 4   # roughly 1% of 200

def test_is_anomaly_percentile_invalid():
    det = DummyDetector().fit(normal_X())
    with pytest.raises(ValueError, match="percentile"):
        det.is_anomaly(normal_X(10), percentile=0.0)


# ---------------------------------------------------------------------------
# score_samples is an alias for decision_function
# ---------------------------------------------------------------------------

def test_score_samples_equals_decision_function():
    X = normal_X(30)
    det = DummyDetector().fit(X)
    np.testing.assert_array_equal(det.score_samples(X), det.decision_function(X))


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
    X = normal_X(50)
    det = DummyDetector(percentile=90.0).fit(X)
    scores_before = det.decision_function(X)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dummy.pkl"
        det.save(path)
        det2 = DummyDetector.load(path)

    scores_after = det2.decision_function(X)
    np.testing.assert_array_almost_equal(scores_before, scores_after)
    assert np.isclose(det2.threshold_, det.threshold_)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

def test_repr_unfitted():
    det = DummyDetector(percentile=90.0)
    r = repr(det)
    assert "DummyDetector" in r
    assert "fitted=False" in r

def test_repr_fitted():
    det = DummyDetector().fit(normal_X())
    assert "fitted=True" in repr(det)
