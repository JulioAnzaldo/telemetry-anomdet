import numpy as np
import pytest

# Distillation reads a fitted torch KANLayer; skip if the deep extra is absent.
pytest.importorskip("torch")

import torch  # noqa: E402

from telemetry_anomdet.models.deep._kan import KANGDNNet, KANLayer  # noqa: E402
from telemetry_anomdet.models.deep.distill import (  # noqa: E402
    KANGDNNetNumpy,
    KANGDNNumpy,
    KANLayerNumpy,
    extract_kan_gdn,
    extract_kan_gdn_net,
    extract_kan_layer,
)
from telemetry_anomdet.models.deep.kan_gdn import KANGDN  # noqa: E402

# ---------------------------------------------------------------------------
# Extraction round-trips to a torch-free evaluator that matches exactly
# ---------------------------------------------------------------------------


def test_extract_returns_torch_free_dict():
    layer = KANLayer(in_features=4, out_features=3, grid_size=5, spline_order=3)
    d = extract_kan_layer(layer)
    assert d["in_features"] == 4
    assert d["out_features"] == 3
    assert d["spline_order"] == 3
    # All arrays are plain NumPy (no torch types leak into the distilled form).
    assert isinstance(d["grid"], np.ndarray)
    assert isinstance(d["base_weight"], np.ndarray)
    assert d["base_weight"].shape == (3, 4)
    assert d["spline_weight"].shape == (3, 4, 5 + 3)


def test_numpy_matches_torch_forward():
    torch.manual_seed(0)
    layer = KANLayer(in_features=5, out_features=2, grid_size=6, spline_order=3)
    # Train a few steps so weights are non-trivial (not just init).
    x = torch.randn(64, 5)
    y = torch.sin(x[:, :2])
    opt = torch.optim.Adam(layer.parameters(), lr=0.02)
    for _ in range(50):
        opt.zero_grad()
        torch.nn.functional.mse_loss(layer(x), y).backward()
        opt.step()

    npl = KANLayerNumpy(extract_kan_layer(layer))

    x_test = torch.randn(20, 5)
    torch_out = layer(x_test).detach().numpy()
    np_out = npl.forward(x_test.numpy())

    assert np_out.shape == (20, 2)
    np.testing.assert_allclose(np_out, torch_out, rtol=1e-4, atol=1e-5)


def test_numpy_preserves_leading_dims():
    layer = KANLayer(in_features=8, out_features=8)
    npl = KANLayerNumpy(extract_kan_layer(layer))
    x = torch.randn(4, 7, 8)
    np.testing.assert_allclose(
        npl.forward(x.numpy()), layer(x).detach().numpy(), rtol=1e-4, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Per-edge 1-D functions sum to the full output
# ---------------------------------------------------------------------------


def test_edge_functions_sum_to_output():
    torch.manual_seed(1)
    layer = KANLayer(in_features=3, out_features=2, grid_size=5, spline_order=3)
    npl = KANLayerNumpy(extract_kan_layer(layer))

    x = np.random.default_rng(0).normal(size=(15, 3))
    full = npl.forward(x)  # (15, 2)

    # Reconstruct output 0 as the sum over its input edges.
    recon0 = sum(npl.edge_function(0, i)(x[:, i]) for i in range(3))
    np.testing.assert_allclose(recon0, full[:, 0], rtol=1e-5, atol=1e-6)


def test_edge_function_is_callable_on_scalars_and_arrays():
    layer = KANLayer(in_features=2, out_features=1)
    npl = KANLayerNumpy(extract_kan_layer(layer))
    phi = npl.edge_function(0, 0)
    assert phi(np.array([0.5])).shape == (1,)
    assert phi(np.linspace(-1, 1, 9)).shape == (9,)


# ---------------------------------------------------------------------------
# Whole-network distillation: GATEncoder + both KAN layers
# ---------------------------------------------------------------------------


def _trained_net(n_nodes=6, window=5, embed_dim=8, topk=3, steps=20):
    """A KANGDNNet nudged off its initialisation so weights are non-trivial."""
    torch.manual_seed(0)
    net = KANGDNNet(n_nodes=n_nodes, window=window, embed_dim=embed_dim, topk=topk)
    x = torch.randn(16, n_nodes, window)
    y = x.mean(dim=-1)
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.mse_loss(net(x), y).backward()
        opt.step()
    net.eval()
    return net


def test_extract_net_returns_torch_free_spec():
    net = _trained_net()
    spec = extract_kan_gdn_net(net)

    assert spec["n_nodes"] == 6
    assert spec["window"] == 5
    assert spec["embed_dim"] == 8
    assert spec["embedding"].shape == (6, 8)
    assert spec["feat_weight"].shape == (8, 5)
    # The attention vector splits into source and target halves (2 * embed_dim).
    assert spec["attn_src"].shape == (16,)
    assert spec["attn_dst"].shape == (16,)
    assert spec["adj"].dtype == bool
    assert isinstance(spec["activation"], dict)
    assert isinstance(spec["out"], dict)


def test_frozen_graph_matches_torch_topk_with_self_loops():
    from telemetry_anomdet.models.deep._net import topk_graph

    net = _trained_net()
    spec = extract_kan_gdn_net(net)

    v = net.encoder.embedding.weight
    expected = topk_graph(v, net.encoder.topk) | torch.eye(6, dtype=torch.bool)
    np.testing.assert_array_equal(spec["adj"], expected.numpy())


def test_numpy_net_matches_torch_forward():
    net = _trained_net()
    npn = KANGDNNetNumpy(extract_kan_gdn_net(net))

    x = torch.randn(7, 6, 5)
    with torch.no_grad():
        torch_out = net(x).numpy()
    np_out = npn.forward(x.numpy())

    assert np_out.shape == (7, 6)
    np.testing.assert_allclose(np_out, torch_out, rtol=1e-4, atol=1e-5)


def test_numpy_net_matches_torch_on_out_of_grid_inputs():
    """Inputs beyond the spline grid fall through to the SiLU base path."""
    net = _trained_net()
    npn = KANGDNNetNumpy(extract_kan_gdn_net(net))

    x = torch.randn(4, 6, 5) * 25.0
    with torch.no_grad():
        torch_out = net(x).numpy()
    np.testing.assert_allclose(npn.forward(x.numpy()), torch_out, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Whole-detector distillation: the deployable artifact
# ---------------------------------------------------------------------------


def _fitted_detector(n_windows=40, window_size=6, n_features=5, **kwargs):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_windows, window_size, n_features))
    det = KANGDN(embed_dim=8, topk=3, epochs=3, batch_size=16, random_state=0, **kwargs)
    det.fit(X)
    return det, X


def test_extract_detector_requires_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        extract_kan_gdn(KANGDN())


def test_distilled_detector_matches_decision_function():
    det, X = _fitted_detector()
    dist = KANGDNNumpy(extract_kan_gdn(det))

    np.testing.assert_allclose(
        dist.decision_function(X), det.decision_function(X), rtol=1e-4, atol=1e-5
    )


def test_distilled_detector_matches_labels_on_unseen_windows():
    det, _ = _fitted_detector()
    dist = KANGDNNumpy(extract_kan_gdn(det))

    X_new = np.random.default_rng(7).normal(size=(15, 6, 5))
    np.testing.assert_array_equal(dist.predict(X_new), det.predict(X_new))


def test_distilled_detector_handles_unscaled_model():
    det, X = _fitted_detector(scale=False)
    spec = extract_kan_gdn(det)
    assert spec["scaler_mean"] is None

    dist = KANGDNNumpy(spec)
    np.testing.assert_allclose(
        dist.decision_function(X), det.decision_function(X), rtol=1e-4, atol=1e-5
    )


def test_distilled_spec_carries_the_threshold():
    det, _ = _fitted_detector()
    spec = extract_kan_gdn(det)
    assert spec["threshold"] == pytest.approx(det.threshold_)
    assert spec["err_median"].shape == (5,)
    assert spec["err_iqr"].shape == (5,)
