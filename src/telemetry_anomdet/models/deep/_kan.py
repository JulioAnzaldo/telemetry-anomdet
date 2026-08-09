# src/telemetry_anomdet/models/deep/_kan.py

"""
A minimal Kolmogorov-Arnold Network (KAN) layer.

This is a hand rolled B-spline KAN layer following Liu et al. (2024) "KAN:
Kolmogorov-Arnold Networks" and the formulation used by Wang et al.'s KGL
(eq. 6-7): each edge carries a learnable univariate function, so an output is a
sum of 1D functions of the inputs rather than a fixed activation on a linear
combination. That structure is what makes a KAN cleanly distillable to symbolic
equations later (each edge spline snaps to a closed form).

Each edge function is ``phi(x) = w_base * silu(x) + w_spline * spline(x)``, where
``spline(x)`` is a B-spline over a fixed grid (the residual base path keeps the
layer for inputs that fall outside the grid). This mirrors the
efficient-kan formulation but is written out here so the toolkit keeps no extra
dependency and the math stays inspectable.

Like ``_net.py``, this module imports torch at the top and is imported lazily by
the detector wrapper, so the base install stays torch-free.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ._net import GATEncoder


class KANLayer(nn.Module):
    """
    A B-spline Kolmogorov-Arnold layer: ``out_j = sum_i phi_ij(x_i)``.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    grid_size : int, default=5
        Number of intervals in the spline grid. More intervals -> more flexible
        (and more spline coefficients per edge).
    spline_order : int, default=3
        B-spline order (3 = cubic).
    grid_range : tuple of float, default=(-2.0, 2.0)
        Range the spline grid spans. Inputs outside this range are handled by
        the residual base (SiLU) path. Standardised inputs mostly fall inside.

    Notes
    -----
    Accepts input of shape ``(..., in_features)`` and returns
    ``(..., out_features)``; leading dimensions are flattened internally, so the
    layer drops in as an activation over ``(batch, n_nodes, embed_dim)`` tensors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Uniform knot vector, extended by spline_order on each side so the
        # recursion has full support at the grid edges. Shared across features.
        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * step
            + grid_range[0]
        )
        grid = grid.expand(in_features, -1).contiguous()  # (in, grid_size + 2*order + 1)
        self.register_buffer("grid", grid)

        # Residual base path (like a standard weight) + spline coefficients.
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.base_fn = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the B-spline bases for each input feature.

        Parameters
        ----------
        x : torch.Tensor, shape (n, in_features)

        Returns
        -------
        bases : torch.Tensor, shape (n, in_features, grid_size + spline_order)
        """
        grid = self.grid  # (in_features, grid_size + 2*order + 1)
        x = x.unsqueeze(-1)  # (n, in_features, 1)

        # Order-0 bases: indicator of which knot span x falls into.
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # Cox-de Boor recursion up to the requested order.
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the KAN layer.

        Parameters
        ----------
        x : torch.Tensor, shape (..., in_features)

        Returns
        -------
        y : torch.Tensor, shape (..., out_features)
        """
        lead_shape = x.shape[:-1]
        x = x.reshape(-1, self.in_features)  # (n, in_features)

        base = self.base_fn(x) @ self.base_weight.t()  # (n, out_features)

        bases = self.b_splines(x)  # (n, in_features, coeff)
        spline = (
            bases.reshape(x.shape[0], -1) @ self.spline_weight.reshape(self.out_features, -1).t()
        )

        y = base + spline  # (n, out_features)
        return y.reshape(*lead_shape, self.out_features)


class KANGDNNet(nn.Module):
    """
    KAN-GAT forecasting network: the shared ``GATEncoder`` with a KAN activation,
    followed by a KAN forecast head.

    Mirrors ``GDNNet`` but replaces the two MLP-style nonlinearities with KAN
    layers (Wang et al.'s KGL): the post-aggregation activation (eq. 3) and the
    per-node output layer. The graph/attention machinery is untouched. Because
    both learnable pieces are KANs, they distil cleanly to symbolic equations.

    Parameters
    ----------
    n_nodes : int
        Number of sensors / feature channels.
    window : int
        Input context length per node (window_size - 1).
    embed_dim : int, default=64
        Embedding / hidden dimensionality.
    topk : int, default=15
        Number of graph neighbours per node.
    grid_size : int, default=5
        Spline grid intervals for the KAN layers.
    spline_order : int, default=3
        B-spline order for the KAN layers.

    Notes
    -----
    Forward input is ``x`` of shape ``(batch, n_nodes, window)`` and the output
    is the one-step forecast of shape ``(batch, n_nodes)``.
    """

    def __init__(
        self,
        n_nodes: int,
        window: int,
        embed_dim: int = 64,
        topk: int = 15,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.window = window
        self.embed_dim = embed_dim
        self.topk = topk

        self.encoder = GATEncoder(
            n_nodes,
            window,
            embed_dim=embed_dim,
            topk=topk,
            activation=KANLayer(
                embed_dim, embed_dim, grid_size=grid_size, spline_order=spline_order
            ),
        )
        self.out = KANLayer(embed_dim, 1, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forecast the next value for every sensor.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_nodes, window)

        Returns
        -------
        pred : torch.Tensor, shape (batch, n_nodes)
        """
        z = self.encoder(x)  # (batch, n_nodes, embed_dim)
        pred = self.out(z).squeeze(-1)  # (batch, n_nodes)
        return pred
