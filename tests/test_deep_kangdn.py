import numpy as np
import pytest

# KANGDN needs the optional deep extra (torch). Skip the module if absent.
pytest.importorskip("torch")

from telemetry_anomdet.models.deep import KANGDN  # noqa: E402
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


def fast_kangdn(**kw):
    kw.setdefault("epochs", 3)
    kw.setdefault("embed_dim", 16)
    kw.setdefault("topk", 3)
    kw.setdefault("grid_size", 4)
    kw.setdefault("random_state", 0)
    return KANGDN(**kw)


# ---------------------------------------------------------------------------
# It is a GDN (inherits the whole pipeline)
# ---------------------------------------------------------------------------


def test_kangdn_is_a_gdn():
    assert issubclass(KANGDN, GDN)


def test_kangdn_builds_kan_network():
    from telemetry_anomdet.models.deep._kan import KANGDNNet

    det = fast_kangdn().fit(make_series(np.random.default_rng(0), 40))
    assert isinstance(det.net, KANGDNNet)


# ---------------------------------------------------------------------------
# Smoke test + post-fit attributes
# ---------------------------------------------------------------------------


def test_kangdn_basic_fit_and_scores():
    rng = np.random.default_rng(0)
    det = fast_kangdn().fit(make_series(rng, 120))
    scores = det.decision_function(make_series(rng, 20))
    assert scores.shape == (20,)
    assert scores.dtype.kind == "f"
    assert np.isfinite(scores).all()


def test_kangdn_postfit_attributes():
    rng = np.random.default_rng(1)
    det = fast_kangdn(percentile=95.0).fit(make_series(rng, 100))
    assert det.decision_scores_.shape == (100,)
    assert isinstance(det.threshold_, float)
    assert set(det.labels_).issubset({0, 1})
    assert det.n_nodes_ == 4
    assert det.window_ == 9


# ---------------------------------------------------------------------------
# Detection + determinism
# ---------------------------------------------------------------------------


def test_kangdn_anomalous_scores_higher():
    rng = np.random.default_rng(3)
    X_train = make_series(rng, 200, scale=1.0)
    X_normal = make_series(rng, 40, scale=1.0)
    X_anom = make_series(rng, 40, scale=1.0, shift=8.0)

    det = fast_kangdn(epochs=8).fit(X_train)
    assert det.decision_function(X_anom).mean() > det.decision_function(X_normal).mean()


def test_kangdn_reproducible_with_seed():
    rng = np.random.default_rng(4)
    X = make_series(rng, 80)
    a = fast_kangdn(random_state=123).fit(X).decision_scores_
    b = fast_kangdn(random_state=123).fit(X).decision_scores_
    np.testing.assert_allclose(a, b)


# ---------------------------------------------------------------------------
# Inherited validation + repr surfaces KAN params
# ---------------------------------------------------------------------------


def test_kangdn_rejects_feature_mismatch():
    rng = np.random.default_rng(7)
    det = fast_kangdn().fit(make_series(rng, 50, n_features=4))
    with pytest.raises(ValueError, match="fitted on 4"):
        det.decision_function(make_series(rng, 10, n_features=5))


def test_kangdn_repr_shows_kan_params():
    r = repr(fast_kangdn(grid_size=7, spline_order=2))
    assert "grid_size=7" in r
    assert "spline_order=2" in r
