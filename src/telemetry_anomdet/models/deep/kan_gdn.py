# src/telemetry_anomdet/models/deep/kan_gdn.py

"""
KAN-GDN: a Graph Deviation Network whose nonlinearities are KAN layers.

Same detector as :class:`~telemetry_anomdet.models.deep.gdn.GDN`. Same graph,
attention, per-channel scaler, forecasting objective, and graph-deviation
scoring, but the post-aggregation activation and the forecast head are
Kolmogorov-Arnold (KAN) layers instead of ReLU + MLP, following Wang et al.'s
KGL (KAN-GAT, eq. 3 and the KAN output layer).

The point of the swap is deployability: a KAN is a sum of learnable 1D spline
functions, so a fitted ``KANGDN`` distils cleanly to closed-form symbolic
equations (via SymTorch / symbolic regression) that can run on a microcontroller
and be audited, unlike GDN's dense MLP head. GDN remains the fast, standard
detector and the ablation baseline; KANGDN is the distillable variant.

torch is an optional dependency; install the deep extra to use this detector::

    uv sync --extra deep
"""

from __future__ import annotations

from .gdn import GDN


class KANGDN(GDN):
    """
    KAN-GAT Graph Deviation Network detector.

    Inherits GDN's full pipeline (input scaling, training loop, deviation
    scoring, thresholding) and only swaps the network architecture: the
    ``GATEncoder`` gets a KAN activation and the forecast head is a KAN layer.

    Parameters
    ----------
    embed_dim, topk, epochs, batch_size, lr, scale, device, random_state, percentile
        Same as :class:`~telemetry_anomdet.models.deep.gdn.GDN`.
    grid_size : int, default=5
        Number of spline grid intervals in the KAN layers. Larger = more
        flexible edge functions (and more coefficients to distil).
    spline_order : int, default=3
        B-spline order for the KAN layers (3 = cubic).

    Attributes (set after fit)
    --------------------------
    Same as GDN, plus ``net`` is a ``KANGDNNet``.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        topk: int = 15,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        scale: bool = True,
        device: str | None = None,
        random_state: int | None = None,
        percentile: float = 95.0,
        grid_size: int = 5,
        spline_order: int = 3,
        smoothing: float | None = None,
        score_channels=None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            topk=topk,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            scale=scale,
            device=device,
            random_state=random_state,
            percentile=percentile,
            smoothing=smoothing,
            score_channels=score_channels,
        )
        self.grid_size = grid_size
        self.spline_order = spline_order

    def _build_net(self):
        from ._kan import KANGDNNet

        return KANGDNNet(
            n_nodes=self.n_nodes_,
            window=self.window_,
            embed_dim=self.embed_dim,
            topk=self.topk,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
        )

    def _get_params(self) -> dict:
        params = super()._get_params()
        params["grid_size"] = self.grid_size
        params["spline_order"] = self.spline_order
        return params
