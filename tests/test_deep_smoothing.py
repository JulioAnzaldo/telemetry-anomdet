import numpy as np
import pytest

pytest.importorskip("torch")

from telemetry_anomdet.models.deep.codegen import generate_c  # noqa: E402
from telemetry_anomdet.models.deep.distill import (  # noqa: E402
    KANGDNNumpy,
    extract_kan_gdn,
)
from telemetry_anomdet.models.deep.gdn import GDN  # noqa: E402
from telemetry_anomdet.models.deep.kan_gdn import KANGDN  # noqa: E402

WINDOW_SIZE = 6
N_FEATURES = 5


def _windows(n=40, seed=0):
    return np.random.default_rng(seed).normal(size=(n, WINDOW_SIZE, N_FEATURES))


# ---------------------------------------------------------------------------
# The EWMA recursion itself
# ---------------------------------------------------------------------------


def test_ewma_matches_the_recursion():
    e = np.array([[1.0], [0.0], [0.0], [0.0]])
    got = GDN.ewma(e, 0.5)
    np.testing.assert_allclose(got[:, 0], [1.0, 0.5, 0.25, 0.125])


def test_ewma_alpha_one_is_identity():
    e = np.random.default_rng(0).normal(size=(10, 3))
    np.testing.assert_allclose(GDN.ewma(e, 1.0), e)


def test_ewma_is_causal():
    """A change at time t must not affect any earlier output."""
    e = np.random.default_rng(1).normal(size=(20, 2)) ** 2
    base = GDN.ewma(e, 0.3)
    bumped = e.copy()
    bumped[10] += 50.0
    after = GDN.ewma(bumped, 0.3)
    np.testing.assert_allclose(after[:10], base[:10])
    assert after[10, 0] > base[10, 0]


def test_ewma_suppresses_an_isolated_spike():
    e = np.ones((30, 1)) * 0.1
    e[15] = 10.0
    smoothed = GDN.ewma(e, 0.2)
    assert smoothed[15, 0] < e[15, 0] / 4.0


def test_rejects_invalid_smoothing():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="smoothing"):
            GDN(smoothing=bad)


# ---------------------------------------------------------------------------
# Smoothing through the detector and the distilled evaluator
# ---------------------------------------------------------------------------


def _fit(smoothing):
    det = KANGDN(embed_dim=8, topk=3, epochs=3, batch_size=16, random_state=0, smoothing=smoothing)
    X = _windows()
    det.fit(X)
    return det, X


def test_smoothing_changes_the_scores():
    plain, X = _fit(None)
    smooth, _ = _fit(0.3)
    assert not np.allclose(plain.decision_function(X), smooth.decision_function(X))


def test_smoothing_is_recorded_in_params():
    det, _ = _fit(0.3)
    assert det._get_params()["smoothing"] == 0.3


def test_distilled_evaluator_reproduces_smoothed_scores():
    det, X = _fit(0.3)
    spec = extract_kan_gdn(det)
    assert spec["smoothing"] == 0.3
    np.testing.assert_allclose(
        KANGDNNumpy(spec).decision_function(X), det.decision_function(X), rtol=1e-4, atol=1e-5
    )


def test_codegen_refuses_a_smoothed_detector():
    """EWMA is stateful across windows; the emitted entry points are not."""
    det, _ = _fit(0.3)
    with pytest.raises(NotImplementedError, match="stateful"):
        generate_c(extract_kan_gdn(det))


def test_codegen_still_accepts_an_unsmoothed_detector():
    det, _ = _fit(None)
    assert "kangdn_score" in generate_c(extract_kan_gdn(det))["kangdn.c"]
