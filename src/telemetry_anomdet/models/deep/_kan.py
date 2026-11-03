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

from ._ar import FEAT_MODES
from ._net import GATEncoder

__all__ = ["ARKANFeatures", "FEAT_MODES", "KANGDNNet", "KANLayer"]


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


class ARKANFeatures(nn.Module):
    """
    Node feature transform with an optional AR memory filter and residual path.

    Implements the non-linear entries of :data:`FEAT_MODES` from two independent
    flags, ``ar`` and ``residual``. The frozen AR filter is held here rather than
    in ``GATEncoder`` so the encoder stays agnostic about what its transform does.

    Parameters
    ----------
    n_nodes : int
        Number of sensors. The AR filter is solved per node.
    window : int
        Input context length.
    embed_dim : int
        Output width.
    grid_size, spline_order : int
        Spline configuration for the KAN branch.
    ar : bool, default=True
        Apply the frozen AR filter to the KAN branch's input.
    residual : bool, default=False
        Add a parallel ``nn.Linear(window, embed_dim)`` over the *unfiltered*
        window.

    Notes
    -----
    ``ar=True, residual=False`` is AR-KAN as published. The filter is diagonal,
    so it scales the window element-wise rather than mixing it, and on strongly
    autocorrelated signals Yule-Walker concentrates almost all of its weight on
    the most recent lag. The KAN downstream then sees close to the previous
    sample alone rather than the window.

    ``residual=True`` keeps a parallel linear branch over the unfiltered window,
    so the AR-KAN branch adds to a baseline instead of replacing it.

    Which combination performs best depends on whether the input's channels are
    genuinely related, and it reverses between datasets. See
    :doc:`/user_guide/feature_transforms` for the measurements and what they
    imply for choosing a mode.
    """

    def __init__(
        self,
        n_nodes: int,
        window: int,
        embed_dim: int,
        grid_size: int = 5,
        spline_order: int = 3,
        ar: bool = True,
        residual: bool = False,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.window = window
        self.embed_dim = embed_dim

        self.kan = KANLayer(window, embed_dim, grid_size=grid_size, spline_order=spline_order)
        self.linear = nn.Linear(window, embed_dim) if residual else None
        # Frozen (a buffer, not a Parameter): ones until set_ar_filters runs, so
        # an unfitted forward pass is not silently scaled by garbage.
        if ar:
            self.register_buffer("ar_filter", torch.ones(n_nodes, window))
        else:
            self.ar_filter = None

    def set_ar_filters(self, filters) -> None:
        """
        Install the frozen AR memory coefficients (AR-KAN stage 1).

        Parameters
        ----------
        filters : array-like, shape (n_nodes, window)
            Per-node coefficients from :func:`._ar.ar_filters`, aligned so the
            last entry multiplies the most recent timestep.

        Raises
        ------
        RuntimeError
            If this transform was built without an AR stage.
        """
        if self.ar_filter is None:
            raise RuntimeError("This feature transform was built without an AR stage.")
        tensor = torch.as_tensor(filters, dtype=self.ar_filter.dtype, device=self.ar_filter.device)
        expected = (self.n_nodes, self.window)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"AR filters must have shape {expected}, got {tuple(tensor.shape)}")
        self.ar_filter.copy_(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_nodes, window)

        Returns
        -------
        h : torch.Tensor, shape (batch, n_nodes, embed_dim)
        """
        filtered = x if self.ar_filter is None else x * self.ar_filter
        h = self.kan(filtered)
        if self.linear is not None:
            h = h + self.linear(x)  # unfiltered: the residual keeps the window
        return h


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
    feat_mode : str, default="linear"
        Node feature transform, one of :data:`FEAT_MODES`. The default keeps the
        single ``nn.Linear`` that both GDN and KANGDN have always used, so the
        splines only ever see an already-compressed embedding. The other modes
        put a nonlinearity on each node's own history, which is the only path
        that does anything at all when ``n_nodes == 1`` (attention over a lone
        self-loop is a no-op).

        Any KAN mode costs roughly ``(grid_size + spline_order)`` times the
        feature transform's coefficients, which dominate the distilled flash
        footprint at small ``embed_dim``. The AR filter itself trains for free
        and costs ``n_nodes * window`` frozen floats.

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
        feat_mode: str = "linear",
    ):
        super().__init__()
        if feat_mode not in FEAT_MODES:
            raise ValueError(f"feat_mode must be one of {FEAT_MODES}, got {feat_mode!r}")
        self.n_nodes = n_nodes
        self.window = window
        self.embed_dim = embed_dim
        self.topk = topk
        self.feat_mode = feat_mode

        feat = None
        if feat_mode != "linear":
            feat = ARKANFeatures(
                n_nodes,
                window,
                embed_dim,
                grid_size=grid_size,
                spline_order=spline_order,
                ar=feat_mode.startswith("ar_kan"),
                residual=feat_mode.endswith("_residual"),
            )
        self.encoder = GATEncoder(
            n_nodes,
            window,
            embed_dim=embed_dim,
            topk=topk,
            activation=KANLayer(
                embed_dim, embed_dim, grid_size=grid_size, spline_order=spline_order
            ),
            feat=feat,
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
