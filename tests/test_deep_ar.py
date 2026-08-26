"""
Tests for the AR-KAN input path: the frozen Yule-Walker filters (`_ar.py`) and
their wiring into KANGDN.

The `_ar` module itself is torch-free NumPy, so those tests run in the base
install. Everything from `TestARKANNetwork` down needs the deep extra.
"""

import warnings

import numpy as np
import pytest

from telemetry_anomdet.models.deep._ar import ar_filters, yule_walker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ar1_segments(rng, phi, n_segments=200, length=32):
    """Segments drawn from a stationary AR(1) process x(n) = phi x(n-1) + e."""
    out = np.zeros((n_segments, length))
    noise = rng.normal(0, 1.0, size=(n_segments, length))
    # Start each segment at its stationary variance so there is no burn-in bias.
    out[:, 0] = noise[:, 0] / np.sqrt(1 - phi**2)
    for t in range(1, length):
        out[:, t] = phi * out[:, t - 1] + noise[:, t]
    return out


# ---------------------------------------------------------------------------
# yule_walker: does it actually recover a known process?
# ---------------------------------------------------------------------------


class TestYuleWalker:
    @pytest.mark.parametrize("phi", [0.9, 0.5, -0.6])
    def test_recovers_ar1_coefficient(self, phi):
        """The whole point: on a known AR(1), a_1 must come back as phi."""
        series = ar1_segments(np.random.default_rng(0), phi)
        coef = yule_walker(series, order=4)
        assert coef[0] == pytest.approx(phi, abs=0.05)
        # Higher lags carry no information in an AR(1) and must stay near zero.
        assert np.abs(coef[1:]).max() < 0.1

    def test_white_noise_gives_near_zero_coefficients(self):
        series = np.random.default_rng(1).normal(size=(200, 32))
        coef = yule_walker(series, order=5)
        assert np.abs(coef).max() < 0.1

    def test_constant_channel_falls_back_to_persistence(self):
        """
        SMAP has 11 channels whose training telemetry never moves. Yule-Walker
        on those is a singular solve; the fallback must be predict-last, not a
        crash or a set of enormous coefficients.
        """
        coef = yule_walker(np.full((10, 20), 3.5), order=6)
        assert coef[0] == 1.0
        assert np.all(coef[1:] == 0.0)

    def test_near_constant_channel_stays_bounded(self):
        """Nine more SMAP channels are near-constant rather than exactly flat."""
        series = np.full((10, 20), 3.5) + np.random.default_rng(2).normal(0, 1e-8, (10, 20))
        coef = yule_walker(series, order=6)
        assert np.isfinite(coef).all()
        assert np.abs(coef).max() < 10.0

    def test_coefficients_are_finite_for_a_trend(self):
        """A ramp is non-stationary; the ridge must keep the solve well posed."""
        ramp = np.tile(np.linspace(0, 10, 24), (5, 1))
        coef = yule_walker(ramp, order=8)
        assert np.isfinite(coef).all()

    def test_rejects_order_below_one(self):
        with pytest.raises(ValueError, match="order must be >= 1"):
            yule_walker(np.zeros((4, 10)), order=0)


# ---------------------------------------------------------------------------
# ar_filters: per-node solve plus the orientation that actually matters
# ---------------------------------------------------------------------------


class TestARFilters:
    def test_shape_matches_context(self):
        ctx = np.random.default_rng(0).normal(size=(50, 3, 12))
        assert ar_filters(ctx).shape == (3, 12)

    def test_newest_timestep_carries_lag_one(self):
        """
        Orientation is the one thing that silently produces a working-but-wrong
        model: the context runs oldest to newest, so a_1 must land on the LAST
        column. On a strong AR(1) that means the last coefficient dominates.

        The default order equals the window, so this is a 16-lag fit to a
        1-lag process on 16-sample segments. It recovers 0.84 rather than 0.90
        because the surplus lags absorb some weight; the tolerance is loose on
        the value and tight on the position, which is what is being asserted.
        """
        ctx = ar1_segments(np.random.default_rng(0), 0.9, n_segments=200, length=16)
        filters = ar_filters(ctx[:, None, :])
        assert filters[0, -1] == pytest.approx(0.9, abs=0.1)
        assert np.abs(filters[0, :-1]).max() < 0.1

    def test_matches_reversed_yule_walker(self):
        ctx = np.random.default_rng(3).normal(size=(40, 2, 10))
        filters = ar_filters(ctx)
        for node in range(2):
            expected = yule_walker(ctx[:, node, :], order=10)[::-1]
            assert filters[node] == pytest.approx(expected)

    def test_shorter_order_zeroes_the_oldest_lags(self):
        ctx = np.random.default_rng(4).normal(size=(40, 2, 10))
        filters = ar_filters(ctx, order=3)
        assert np.all(filters[:, :-3] == 0.0)
        assert np.any(filters[:, -3:] != 0.0)

    def test_channels_are_solved_independently(self):
        """One dead channel must not disturb its neighbours' coefficients."""
        rng = np.random.default_rng(5)
        live = ar1_segments(rng, 0.8, n_segments=100, length=16)
        ctx = np.stack([live, np.full_like(live, 2.0)], axis=1)  # (n, 2, 16)

        filters = ar_filters(ctx)
        assert filters[0, -1] == pytest.approx(0.8, abs=0.06)
        assert filters[1, -1] == 1.0
        assert np.all(filters[1, :-1] == 0.0)

    @pytest.mark.parametrize("order", [0, 11])
    def test_rejects_out_of_range_order(self, order):
        ctx = np.zeros((4, 2, 10))
        with pytest.raises(ValueError, match="order must lie in"):
            ar_filters(ctx, order=order)

    def test_rejects_non_3d_context(self):
        with pytest.raises(ValueError, match="context must be 3D"):
            ar_filters(np.zeros((4, 10)))


# ---------------------------------------------------------------------------
# Wiring into KANGDN
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")

from telemetry_anomdet.models.deep import KANGDN  # noqa: E402
from telemetry_anomdet.models.deep._ar import FEAT_MODES  # noqa: E402
from telemetry_anomdet.models.deep._kan import ARKANFeatures, KANGDNNet  # noqa: E402

# Every mode except the default linear one, i.e. those that install an
# ARKANFeatures transform.
NONLINEAR_MODES = tuple(m for m in FEAT_MODES if m != "linear")
AR_MODES = ("ar_kan", "ar_kan_residual")


def make_series(rng, n_windows, window_size=10, n_features=4):
    base = rng.normal(0, 1.0, size=(n_windows, 1, n_features))
    noise = rng.normal(0, 0.05, size=(n_windows, window_size, n_features))
    ramp = np.linspace(0, 1, window_size)[None, :, None]
    return base + ramp * base * 0.1 + noise


def fast_kangdn(**kw):
    kw.setdefault("epochs", 3)
    kw.setdefault("embed_dim", 8)
    kw.setdefault("topk", 3)
    kw.setdefault("grid_size", 4)
    kw.setdefault("random_state", 0)
    return KANGDN(**kw)


class TestFeatModes:
    def test_linear_is_the_default(self):
        net = KANGDNNet(n_nodes=3, window=9)
        assert net.feat_mode == "linear"
        assert isinstance(net.encoder.feat, torch.nn.Linear)

    @pytest.mark.parametrize("mode", NONLINEAR_MODES)
    def test_nonlinear_modes_install_arkan_features(self, mode):
        net = KANGDNNet(n_nodes=3, window=9, feat_mode=mode)
        assert isinstance(net.encoder.feat, ARKANFeatures)

    def test_kan_mode_has_no_filter_and_no_residual(self):
        """`kan` is the ablation that isolates the nonlinearity from the AR
        filter. If either of these appears, the two are confounded again."""
        feat = KANGDNNet(n_nodes=3, window=9, feat_mode="kan").encoder.feat
        assert feat.ar_filter is None
        assert feat.linear is None

    def test_kan_residual_is_not_offered(self):
        """
        Linear(x) + KAN(x) was measured across SMAP, MSL and ESA-ADB and was
        never best in any regime, tying the plain linear transform it exists to
        improve on. It is not a mode. ARKANFeatures still accepts the flag
        combination, since ar and residual are independent.
        """
        assert "kan_residual" not in FEAT_MODES
        with pytest.raises(ValueError, match="feat_mode must be one of"):
            KANGDNNet(n_nodes=3, window=9, feat_mode="kan_residual")

    def test_each_mode_is_a_distinct_transform(self):
        """No two modes may build the same (ar, residual) combination, or one of
        them is dead weight in the benchmark."""
        seen = {
            (
                (feat := KANGDNNet(n_nodes=3, window=9, feat_mode=mode).encoder.feat).ar_filter
                is not None,
                feat.linear is not None,
            )
            for mode in NONLINEAR_MODES
        }
        assert len(seen) == len(NONLINEAR_MODES)

    def test_ar_kan_filters_but_does_not_keep_a_linear_path(self):
        feat = KANGDNNet(n_nodes=3, window=9, feat_mode="ar_kan").encoder.feat
        assert feat.ar_filter.shape == (3, 9)
        assert feat.linear is None

    def test_residual_keeps_both_branches(self):
        feat = KANGDNNet(n_nodes=3, window=9, feat_mode="ar_kan_residual").encoder.feat
        assert feat.ar_filter.shape == (3, 9)
        assert isinstance(feat.linear, torch.nn.Linear)

    @pytest.mark.parametrize("mode", FEAT_MODES)
    def test_forward_shape_is_the_same_in_every_mode(self, mode):
        net = KANGDNNet(n_nodes=3, window=9, feat_mode=mode)
        assert net(torch.randn(5, 3, 9)).shape == (5, 3)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="feat_mode must be one of"):
            KANGDNNet(n_nodes=3, window=9, feat_mode="arkan")


class TestARFilterWiring:
    @pytest.mark.parametrize("mode", AR_MODES)
    def test_filter_starts_as_a_no_op(self, mode):
        """Ones until fit installs the real coefficients, so an unfitted
        forward pass is not silently scaled by garbage."""
        feat = KANGDNNet(n_nodes=3, window=9, feat_mode=mode).encoder.feat
        assert torch.equal(feat.ar_filter, torch.ones(3, 9))

    def test_filter_is_not_trainable(self):
        """
        A buffer, not a Parameter. If this ever flips, the AR stage stops being
        the closed-form filter the architecture is justified by and the flash
        cost argument changes.
        """
        net = KANGDNNet(n_nodes=3, window=9, feat_mode="ar_kan")
        assert "encoder.feat.ar_filter" in dict(net.named_buffers())
        assert "encoder.feat.ar_filter" not in dict(net.named_parameters())

    def test_set_ar_filters_rejects_wrong_shape(self):
        feat = KANGDNNet(n_nodes=3, window=9, feat_mode="ar_kan").encoder.feat
        with pytest.raises(ValueError, match="AR filters must have shape"):
            feat.set_ar_filters(np.ones((3, 5)))

    def test_set_ar_filters_refuses_without_an_ar_stage(self):
        feat = KANGDNNet(n_nodes=3, window=9, feat_mode="kan").encoder.feat
        with pytest.raises(RuntimeError, match="without an AR stage"):
            feat.set_ar_filters(np.ones((3, 9)))

    def test_zeroing_the_filter_silences_the_kan_branch(self):
        """Direct evidence the multiply sits on the KAN branch's input: with a
        zero filter and no residual, the output cannot depend on x."""
        net = KANGDNNet(n_nodes=2, window=6, embed_dim=4, topk=1, feat_mode="ar_kan")
        net.eval()
        net.encoder.feat.set_ar_filters(np.zeros((2, 6)))
        with torch.no_grad():
            a = net(torch.randn(4, 2, 6))
            b = net(torch.randn(4, 2, 6))
        assert torch.allclose(a, b)

    def test_residual_survives_a_zero_filter(self):
        """The point of the residual mode: the linear branch sees the unfiltered
        window, so killing the AR branch must not make the model constant."""
        net = KANGDNNet(n_nodes=2, window=6, embed_dim=4, topk=1, feat_mode="ar_kan_residual")
        net.eval()
        net.encoder.feat.set_ar_filters(np.zeros((2, 6)))
        with torch.no_grad():
            a = net(torch.randn(4, 2, 6))
            b = net(torch.randn(4, 2, 6))
        assert not torch.allclose(a, b)


class TestFeatModeDetector:
    @pytest.mark.parametrize("mode", AR_MODES)
    def test_fit_installs_solved_filters(self, mode):
        det = fast_kangdn(feat_mode=mode).fit(make_series(np.random.default_rng(0), 60))
        assert det.ar_filters_.shape == (4, 9)
        # Ones would mean _init_net_from_data never ran.
        assert not np.allclose(det.ar_filters_, 1.0)
        assert torch.allclose(
            det.net.encoder.feat.ar_filter.cpu(),
            torch.as_tensor(det.ar_filters_, dtype=torch.float32),
        )

    @pytest.mark.parametrize("mode", ["linear", "kan"])
    def test_no_filters_without_an_ar_stage(self, mode):
        det = fast_kangdn(feat_mode=mode).fit(make_series(np.random.default_rng(0), 60))
        assert det.ar_filters_ is None

    @pytest.mark.parametrize("mode", FEAT_MODES)
    def test_scores_are_finite(self, mode):
        rng = np.random.default_rng(0)
        det = fast_kangdn(feat_mode=mode).fit(make_series(rng, 80))
        scores = det.decision_function(make_series(rng, 20))
        assert scores.shape == (20,)
        assert np.isfinite(scores).all()

    @pytest.mark.parametrize("mode", NONLINEAR_MODES)
    def test_single_channel_fits(self, mode):
        """
        The case these modes exist for: one node, one self-loop, no graph signal
        at all. Must train and produce varying scores.
        """
        rng = np.random.default_rng(1)
        X = make_series(rng, 120, window_size=12, n_features=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            det = fast_kangdn(feat_mode=mode, topk=1).fit(X)
            scores = det.decision_function(make_series(rng, 40, window_size=12, n_features=1))
        assert np.isfinite(scores).all()
        assert scores.std() > 0

    def test_params_round_trip(self):
        params = KANGDN(feat_mode="ar_kan", ar_order=7)._get_params()
        assert params["feat_mode"] == "ar_kan"
        assert params["ar_order"] == 7

    def test_bad_mode_fails_at_construction_not_at_fit(self):
        with pytest.raises(ValueError, match="feat_mode must be one of"):
            KANGDN(feat_mode="ar-kan")

    def test_rejects_bad_ar_order(self):
        with pytest.raises(ValueError, match="ar_order must be >= 1"):
            KANGDN(ar_order=0)

    @pytest.mark.parametrize("mode", FEAT_MODES)
    def test_is_deterministic(self, mode):
        X = make_series(np.random.default_rng(2), 60)
        a = fast_kangdn(feat_mode=mode).fit(X).decision_scores_
        b = fast_kangdn(feat_mode=mode).fit(X).decision_scores_
        assert np.array_equal(a, b)


class TestFeatModeDistillation:
    @pytest.mark.parametrize("mode", FEAT_MODES)
    def test_numpy_matches_torch(self, mode):
        """
        The distilled NumPy evaluator must reproduce every mode exactly,
        otherwise the deployable artifact is a different model from the one the
        benchmark measured.
        """
        from telemetry_anomdet.models.deep.distill import KANGDNNetNumpy, extract_kan_gdn_net

        rng = np.random.default_rng(0)
        det = fast_kangdn(feat_mode=mode).fit(make_series(rng, 60))
        det.net.eval()

        x = make_series(rng, 8)[:, :-1, :].transpose(0, 2, 1)  # (batch, nodes, window)
        with torch.no_grad():
            expected = det.net(torch.as_tensor(x, dtype=torch.float32)).numpy()
        got = KANGDNNetNumpy(extract_kan_gdn_net(det.net)).forward(x)
        assert got == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize(
        "mode, has_linear, has_kan, has_ar",
        [
            ("linear", True, False, False),
            ("kan", False, True, False),
            ("ar_kan", False, True, True),
            ("ar_kan_residual", True, True, True),
        ],
    )
    def test_extraction_populates_the_right_branches(self, mode, has_linear, has_kan, has_ar):
        from telemetry_anomdet.models.deep.distill import extract_kan_gdn_net

        det = fast_kangdn(feat_mode=mode).fit(make_series(np.random.default_rng(0), 60))
        extracted = extract_kan_gdn_net(det.net)

        assert (extracted["feat_weight"] is not None) is has_linear
        assert (extracted["feat_bias"] is not None) is has_linear
        assert (extracted["feat_kan"] is not None) is has_kan
        assert (extracted["ar_filter"] is not None) is has_ar
        if has_ar:
            assert extracted["ar_filter"] == pytest.approx(det.ar_filters_, abs=1e-6)

    def test_linear_extraction_keeps_its_shape(self):
        """The default path must keep its old keys so existing golden vectors
        and the C emitter still apply."""
        from telemetry_anomdet.models.deep.distill import extract_kan_gdn_net

        det = fast_kangdn().fit(make_series(np.random.default_rng(0), 60))
        assert extract_kan_gdn_net(det.net)["feat_weight"].shape == (8, 9)

    @pytest.mark.parametrize("mode", NONLINEAR_MODES)
    def test_codegen_refuses_rather_than_emitting_wrong_c(self, mode):
        from telemetry_anomdet.models.deep.codegen import generate_c
        from telemetry_anomdet.models.deep.distill import extract_kan_gdn

        det = fast_kangdn(feat_mode=mode).fit(make_series(np.random.default_rng(0), 60))
        with pytest.raises(NotImplementedError, match="feat_mode='linear'"):
            generate_c(extract_kan_gdn(det))
