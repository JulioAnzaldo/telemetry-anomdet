import numpy as np
import pytest
from telemetry_anomdet.models.ensemble import AnomalyEnsemble
from telemetry_anomdet.models.unsupervised.pca import PCAAnomaly
from telemetry_anomdet.models.unsupervised.kmeans import KMeansAnomaly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
X_TRAIN = RNG.normal(0, 1, size=(100, 10, 4))
X_TEST  = RNG.normal(0, 1, size=(20, 10, 4))


def make_ensemble(**kwargs) -> AnomalyEnsemble:
    return AnomalyEnsemble(
        models={"pca": PCAAnomaly(), "kmeans": KMeansAnomaly(n_clusters=3)},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_ensemble_fit_and_decision_function():
    ens = make_ensemble().fit(X_TRAIN)
    scores = ens.decision_function(X_TEST)
    assert scores.shape == (20,)
    assert np.isfinite(scores).all()


# ---------------------------------------------------------------------------
# Post-fit attributes
# ---------------------------------------------------------------------------

def test_ensemble_postfit_attributes():
    ens = make_ensemble(percentile=95.0).fit(X_TRAIN)
    assert ens.decision_scores_ is not None
    assert ens.decision_scores_.shape == (X_TRAIN.shape[0],)
    assert isinstance(ens.threshold_, float)
    assert ens.labels_ is not None
    assert set(ens.labels_).issubset({0, 1})


def test_ensemble_norm_stats_populated():
    ens = make_ensemble().fit(X_TRAIN)
    assert set(ens._norm_stats.keys()) == {"pca", "kmeans"}


# ---------------------------------------------------------------------------
# score_components
# ---------------------------------------------------------------------------

def test_score_components_keys_and_shapes():
    ens = make_ensemble().fit(X_TRAIN)
    components = ens.score_components(X_TEST)
    assert set(components.keys()) == {"pca", "kmeans"}
    for name, arr in components.items():
        assert arr.shape == (20,), f"{name} has wrong shape"
        assert np.isfinite(arr).all()


# ---------------------------------------------------------------------------
# combine strategies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("combine", ["mean", "median", "max"])
def test_combine_strategies(combine):
    ens = make_ensemble(combine=combine).fit(X_TRAIN)
    scores = ens.decision_function(X_TEST)
    assert scores.shape == (20,)
    assert np.isfinite(scores).all()


def test_combine_invalid_raises():
    with pytest.raises(ValueError, match="combine"):
        make_ensemble(combine="bad").fit(X_TRAIN)


# ---------------------------------------------------------------------------
# normalize strategies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("normalize", ["robust", "minmax", "none"])
def test_normalize_strategies(normalize):
    ens = make_ensemble(normalize=normalize).fit(X_TRAIN)
    scores = ens.decision_function(X_TEST)
    assert scores.shape == (20,)
    assert np.isfinite(scores).all()


def test_normalize_invalid_raises():
    with pytest.raises(ValueError, match="normalization"):
        make_ensemble(normalize="bad").fit(X_TRAIN)


def test_normalize_robust_scores_clipped():
    """Robust-normalized scores should be clipped to [0, 1]."""
    ens = make_ensemble(normalize="robust").fit(X_TRAIN)
    scores = ens.decision_function(X_TEST)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_normalize_minmax_scores_clipped():
    ens = make_ensemble(normalize="minmax").fit(X_TRAIN)
    scores = ens.decision_function(X_TEST)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


# ---------------------------------------------------------------------------
# predict and is_anomaly are inherited from BaseDetector
# ---------------------------------------------------------------------------

def test_ensemble_predict_returns_binary():
    ens = make_ensemble().fit(X_TRAIN)
    preds = ens.predict(X_TEST)
    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1})


def test_ensemble_is_anomaly_threshold_override():
    ens = make_ensemble().fit(X_TRAIN)
    assert ens.is_anomaly(X_TEST, threshold=-999.0).all()
    assert not ens.is_anomaly(X_TEST, threshold=999.0).any()


# ---------------------------------------------------------------------------
# Unfitted ensemble raises on decision_function
# ---------------------------------------------------------------------------

def test_ensemble_unfitted_raises():
    ens = make_ensemble()
    with pytest.raises(RuntimeError, match="not fitted"):
        ens.decision_function(X_TEST)


# ---------------------------------------------------------------------------
# normalize=False flag on decision_function
# ---------------------------------------------------------------------------

def test_decision_function_normalize_false_returns_raw():
    """normalize=False on decision_function skips per-model norm."""
    ens = make_ensemble(normalize="robust").fit(X_TRAIN)
    raw   = ens.decision_function(X_TEST, normalize=False)
    normd = ens.decision_function(X_TEST, normalize=True)
    # They differ when normalization is non-trivial
    assert not np.allclose(raw, normd)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

def test_ensemble_repr():
    ens = make_ensemble()
    assert "fitted=False" in repr(ens)
    ens.fit(X_TRAIN)
    assert "fitted=True" in repr(ens)
    assert "pca" in repr(ens)
    assert "kmeans" in repr(ens)
