import numpy as np
import pytest

# Distillation reads a fitted torch KANLayer; skip if the deep extra is absent.
pytest.importorskip("torch")

import torch  # noqa: E402

from telemetry_anomdet.models.deep._kan import KANLayer  # noqa: E402
from telemetry_anomdet.models.deep.distill import (  # noqa: E402
    KANLayerNumpy,
    extract_kan_layer,
)

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
