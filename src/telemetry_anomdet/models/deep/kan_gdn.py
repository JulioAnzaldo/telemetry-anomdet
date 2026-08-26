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

    ``feat_mode`` additionally swaps the input path, which is the only path that
    does anything on single-channel telemetry: with one node the graph is a lone
    self-loop, so the default linear transform leaves the model with no
    nonlinear view of its own history at all.

    Which mode performs best depends on whether the input's channels are
    genuinely related, and it reverses between datasets, so it is a measurement
    rather than a default. See :doc:`/user_guide/feature_transforms` for the
    comparison, the flash cost of each, and the recommendation.

    Any KAN mode multiplies the feature transform's coefficients by roughly
    ``grid_size + spline_order``, which dominates the distilled flash footprint
    at small ``embed_dim``. Distillation supports every mode; C generation
    supports only ``linear``.

    Parameters
    ----------
    embed_dim, topk, epochs, batch_size, lr, scale, device, random_state, percentile
        Same as :class:`~telemetry_anomdet.models.deep.gdn.GDN`.
    grid_size : int, default=5
        Number of spline grid intervals in the KAN layers. Larger = more
        flexible edge functions (and more coefficients to distil).
    spline_order : int, default=3
        B-spline order for the KAN layers (3 = cubic).
    feat_mode : str, default="linear"
        Node feature transform, one of :data:`~telemetry_anomdet.models.deep._ar.FEAT_MODES`.
        ``linear`` is a single ``nn.Linear`` over the window, ``kan`` is
        ``KAN(x)``, ``ar_kan`` is AR-KAN as published (Wu et al. 2025,
        arXiv:2509.02967) at ``KAN(a * x)``, and ``ar_kan_residual`` is
        ``Linear(x) + KAN(a * x)``. See above for what each costs and buys.
    ar_order : int or None, default=None
        AR order for the ``ar_kan`` modes. Defaults to the full context length,
        one coefficient per timestep. A smaller order zeroes the oldest lags and
        is better conditioned: a full-length fit estimates its top lags from very
        few sample pairs.

    Attributes
    ----------
    net : KANGDNNet
        The fitted torch forecasting network. Other post-fit attributes are the
        same as GDN's.
    ar_filters_ : np.ndarray or None
        Frozen per-channel Yule-Walker coefficients, shape
        ``(n_nodes_, window_)``. None unless ``feat_mode`` includes an AR stage.
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
        feat_mode: str = "linear",
        ar_order: int | None = None,
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
        if ar_order is not None and ar_order < 1:
            raise ValueError(f"ar_order must be >= 1, got {ar_order}")
        # Validated here rather than only at fit time: a typo should fail when
        # the detector is constructed, not after a training run.
        from ._ar import FEAT_MODES

        if feat_mode not in FEAT_MODES:
            raise ValueError(f"feat_mode must be one of {FEAT_MODES}, got {feat_mode!r}")
        self.feat_mode = feat_mode
        self.ar_order = ar_order

        # Fit artifact: set in _init_net_from_data when feat_mode has an AR stage.
        self.ar_filters_ = None

    def _build_net(self):
        from ._kan import KANGDNNet

        return KANGDNNet(
            n_nodes=self.n_nodes_,
            window=self.window_,
            embed_dim=self.embed_dim,
            topk=self.topk,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            feat_mode=self.feat_mode,
        )

    def _init_net_from_data(self, context):
        """
        Solve the frozen AR memory filters and install them in the transform.

        Runs before the optimiser exists, so the coefficients never receive a
        gradient. They come from the scaled training context only, which keeps
        them on the same leakage-free footing as the per-channel scaler.
        """
        if not self.feat_mode.startswith("ar_kan"):
            return

        from ._ar import ar_filters

        self.ar_filters_ = ar_filters(context, order=self.ar_order)
        self.net.encoder.feat.set_ar_filters(self.ar_filters_)

    def _get_params(self) -> dict:
        params = super()._get_params()
        params["grid_size"] = self.grid_size
        params["spline_order"] = self.spline_order
        params["feat_mode"] = self.feat_mode
        params["ar_order"] = self.ar_order
        return params
