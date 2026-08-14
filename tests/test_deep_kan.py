import numpy as np
import pytest

# KAN layer needs the optional deep extra (torch). Skip the module if absent.
pytest.importorskip("torch")

import torch  # noqa: E402

from telemetry_anomdet.models.deep._kan import KANLayer  # noqa: E402

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_kan_layer_2d_shape():
    layer = KANLayer(in_features=3, out_features=2)
    y = layer(torch.randn(10, 3))
    assert y.shape == (10, 2)
    assert torch.isfinite(y).all()


def test_kan_layer_preserves_leading_dims():
    # Drop-in over (batch, n_nodes, embed_dim) tensors: only the last dim maps.
    layer = KANLayer(in_features=8, out_features=8)
    y = layer(torch.randn(4, 5, 8))
    assert y.shape == (4, 5, 8)


def test_kan_bspline_basis_shape():
    layer = KANLayer(in_features=3, out_features=2, grid_size=5, spline_order=3)
    bases = layer.b_splines(torch.randn(7, 3))
    # grid_size + spline_order coefficients per input feature.
    assert bases.shape == (7, 3, 5 + 3)
    assert torch.isfinite(bases).all()


# ---------------------------------------------------------------------------
# The layer actually learns a nonlinear function (the real validation)
# ---------------------------------------------------------------------------


def test_kan_layer_fits_nonlinear_function():
    torch.manual_seed(0)
    layer = KANLayer(in_features=1, out_features=1, grid_size=8, spline_order=3)

    x = torch.linspace(-2, 2, 256).unsqueeze(-1)
    y = torch.sin(2.0 * x)  # smooth nonlinearity within the grid range

    opt = torch.optim.Adam(layer.parameters(), lr=0.02)
    loss_fn = torch.nn.MSELoss()

    initial = loss_fn(layer(x), y).item()
    for _ in range(500):
        opt.zero_grad()
        loss = loss_fn(layer(x), y)
        loss.backward()
        opt.step()
    final = loss_fn(layer(x), y).item()

    # A single cubic-spline edge should approximate sin(2x) closely.
    assert final < 0.01, f"KAN failed to fit sin(2x): final MSE {final:.4f}"
    assert final < initial * 0.1


def test_kan_layer_gradients_flow():
    layer = KANLayer(in_features=4, out_features=3)
    out = layer(torch.randn(6, 4)).sum()
    out.backward()
    assert layer.base_weight.grad is not None
    assert layer.spline_weight.grad is not None
    assert torch.isfinite(layer.spline_weight.grad).all()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_kan_layer_reproducible_with_seed():
    torch.manual_seed(42)
    a = KANLayer(3, 2)(torch.ones(5, 3))
    torch.manual_seed(42)
    b = KANLayer(3, 2)(torch.ones(5, 3))
    np.testing.assert_allclose(a.detach().numpy(), b.detach().numpy())
