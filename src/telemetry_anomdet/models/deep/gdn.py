# src/telemetry_anomdet/models/deep/gdn.py

"""
GDN (Graph Deviation Network) anomaly detector.

A deep, sequence-aware detector that learns a graph over sensor channels and
forecasts each channel one step ahead. Anomaly scores are the *graph deviation
score*: the maximum, over sensors, of each sensor's forecast error normalised by
its training-error statistics.

Reference: Deng & Hooi, "Graph Neural Network-Based Anomaly Detection in
Multivariate Time Series", AAAI 2021.

Unlike the classical detectors, GDN is a *sequence* detector: it consumes the 3D
windowed tensor ``(n_windows, window_size, n_features)`` directly rather than
flattening it via ``features_stat()``. The first ``window_size - 1`` timesteps of
each window are the forecasting context and the final timestep is the target.

torch is an optional dependency. Install the deep extra to use this detector::

    uv sync --extra deep
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..base import BaseDetector


def _import_torch():
    """
    Import torch, raising a friendly error if the deep extra is not installed.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "GDN requires PyTorch, which is an optional dependency. "
            "Install it with the deep extra:\n\n    uv sync --extra deep\n\n"
            "or:  pip install 'telemetry-anomdet[deep]'"
        ) from exc
    return torch


class GDN(BaseDetector):
    """
    Graph Deviation Network detector.

    Parameters
    ----------
    embed_dim : int, default=64
        Dimensionality of the learned per-sensor embeddings and hidden features.
    topk : int, default=15
        Number of graph neighbours retained per sensor. Clamped internally to
        ``n_features - 1``.
    epochs : int, default=30
        Number of training epochs.
    batch_size : int, default=64
        Minibatch size for training.
    lr : float, default=1e-3
        Adam learning rate.
    scale : bool, default=True
        Standardise each channel (per-feature z-score) before training and
        scoring. Recommended: unlike the classical detectors, the network has
        no implicit scale invariance, so unscaled telemetry lets large-magnitude
        channels dominate the forecast. The scaler is fitted on the training
        windows only and reused at inference, so no test statistics leak in.
    device : str or None, default=None
        Torch device string (e.g. "cuda", "cpu"). If None, uses CUDA when
        available, otherwise CPU.
    random_state : int or None, default=None
        Seed for torch and numpy RNGs, for reproducible training.
    percentile : float, default=95.0
        Percentile of training deviation scores used to set ``threshold_``.
    score_channels : sequence of int, optional
        Which feature channels may raise an alarm. Every channel still feeds
        the graph and the forecast; this restricts only the deviation score.
        Defaults to all of them.

        The distinction matters whenever a record mixes continuous sensors with
        discrete status, mode or command channels. A discrete channel produces a
        large forecast error every time it switches, which is a state change
        rather than a fault, so letting it into the score raises an alarm on
        normal operation. Such channels are still worth feeding to the model,
        because they carry context that improves the forecast.

        Choosing them well can outweigh any thresholding decision; see the
        anomaly scoring page of the documentation for the measured effect.

    Attributes (set after fit)
    --------------------------
    decision_scores_ : np.ndarray, shape (n_windows,)
        Graph deviation scores on the training data.
    threshold_ : float
        Default anomaly cutoff derived from training scores at ``percentile``.
    labels_ : np.ndarray, shape (n_windows,)
        Binary anomaly labels on training data. 0 = normal, 1 = anomaly.
    net : GDNNet
        The fitted torch forecasting network.
    n_nodes_ : int
        Number of sensor channels seen during fit.
    window_ : int
        Forecasting context length (window_size - 1).
    scaler : sklearn.preprocessing.StandardScaler or None
        Fitted per-channel scaler when ``scale = True``, otherwise None.

    Notes
    -----
    The internal torch network is kept deliberately separate from this wrapper
    (see ``_net.GDNNet``) so it can be consumed on its own by later
    explainability or symbolic-distillation work without the detector plumbing.
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
        smoothing: float | None = None,
        score_channels: Sequence[int] | None = None,
    ):
        super().__init__(percentile=percentile)
        self.embed_dim = embed_dim
        self.topk = topk
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scale = scale
        self.device = device
        self.random_state = random_state
        if smoothing is not None and not (0.0 < smoothing <= 1.0):
            raise ValueError(f"smoothing must lie in (0, 1], got {smoothing}")
        self.smoothing = smoothing
        if score_channels is not None:
            score_channels = [int(c) for c in score_channels]
            if not score_channels:
                raise ValueError("score_channels must not be empty")
            if len(set(score_channels)) != len(score_channels):
                raise ValueError(f"score_channels contains duplicates: {score_channels}")
            if any(c < 0 for c in score_channels):
                raise ValueError(f"score_channels must be non-negative: {score_channels}")
        self.score_channels = score_channels

        # Fit artifacts: set in fit()
        self.net = None
        self.scaler: StandardScaler | None = None
        self.n_nodes_: int | None = None
        self.window_: int | None = None
        self._err_median_: np.ndarray | None = None
        self._err_iqr_: np.ndarray | None = None
        self._degenerate_: np.ndarray = np.empty(0, dtype=int)

    # ---- helpers ----
    def _resolve_device(self, torch):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _scale_fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit a per-channel standardiser on the training windows and apply it.

        Channels (features) are the graph nodes, so standardisation is per
        feature across every window and timestep. The 3D tensor is reshaped to
        ``(n_windows * window_size, n_features)`` for the fit, then restored.
        Constant channels (e.g. inactive command one-hots) are safe: sklearn's
        StandardScaler maps zero variance to a scale of 1.0.
        """
        if not self.scale:
            self.scaler = None
            return X
        n, w, f = X.shape
        self.scaler = StandardScaler()
        flat = self.scaler.fit_transform(X.reshape(n * w, f))
        return flat.reshape(n, w, f)

    def _scale_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted per-channel standardiser (identity when scale=False)."""
        if not self.scale:
            return X
        if self.scaler is None:
            raise RuntimeError("Scaler is not fitted. Was the model fitted with scale = True?")
        n, w, f = X.shape
        flat = self.scaler.transform(X.reshape(n * w, f))
        return flat.reshape(n, w, f)

    @staticmethod
    def _split_context_target(X: np.ndarray):
        """
        Split windows into forecasting context and target.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)

        Returns
        -------
        context : np.ndarray, shape (n_windows, n_features, window_size - 1)
            Per-node input sequences (transposed so nodes lead the sequence).
        target : np.ndarray, shape (n_windows, n_features)
            Final-timestep value for each node.
        """
        context = np.transpose(X[:, :-1, :], (0, 2, 1))  # (n, n_features, w-1)
        target = X[:, -1, :]  # (n, n_features)
        return context, target

    def _forecast_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Per-window, per-node absolute forecast error, shape (n_windows, n_nodes).
        """
        torch = _import_torch()
        context, target = self._split_context_target(X)
        device = next(self.net.parameters()).device

        self.net.eval()
        errors = []
        with torch.no_grad():
            for start in range(0, context.shape[0], self.batch_size):
                xb = torch.as_tensor(
                    context[start : start + self.batch_size], dtype=torch.float32, device=device
                )
                pred = self.net(xb).cpu().numpy()
                tb = target[start : start + self.batch_size]
                errors.append(np.abs(pred - tb))
        stacked = np.concatenate(errors, axis=0)
        # Smoothing lives here so that the training statistics in fit() and the
        # scores at inference are derived from the same signal.
        if self.smoothing is not None:
            stacked = self.ewma(stacked, self.smoothing)
        return stacked

    @staticmethod
    def ewma(errors: np.ndarray, alpha: float) -> np.ndarray:
        """
        Exponentially weighted moving average down the window (time) axis.

        ``s_t = alpha * e_t + (1 - alpha) * s_{t-1}``, seeded with ``s_0 = e_0``
        and applied independently per node. Smaller ``alpha`` means heavier
        smoothing.

        The recursion only looks backwards, so no future information reaches a
        window's score and the training statistics stay leakage free. Smoothing
        suppresses single-window spikes, which are the dominant source of
        isolated false alarms in a forecasting detector.

        Parameters
        ----------
        errors : np.ndarray, shape (n_windows, n_nodes)
            Per-window, per-node errors, ordered in time.
        alpha : float
            Smoothing factor in (0, 1]. A value of 1.0 is the identity.

        Returns
        -------
        np.ndarray
            Smoothed errors, same shape as the input.
        """
        errors = np.asarray(errors, dtype=float)
        if alpha >= 1.0 or errors.shape[0] < 2:
            return errors
        out = np.empty_like(errors)
        out[0] = errors[0]
        for t in range(1, errors.shape[0]):
            out[t] = alpha * errors[t] + (1.0 - alpha) * out[t - 1]
        return out

    # Lower bound on a channel's training error spread, as a fraction of that
    # channel's own median error.
    #
    # A channel that never moved while training has an error IQR of zero, and
    # dividing by it produces a number decided by the guard epsilon rather than
    # by the data: 1e10 on SMAP, which then wins the per-window maximum and
    # decides every score. Status and mode channels that sit constant for long
    # stretches are the usual source.
    #
    # The ratio is set below the smallest spread any real channel shows, so the
    # floor only ever reaches channels with no measurable spread and never
    # reshapes one that has some. On SMAP the healthy channels run from 0.030 to
    # 2.5 by this measure, so 0.02 catches the three degenerate channels and
    # nothing else, and their deviations land at the same order as the most
    # extreme healthy channel instead of a million times above it.
    _SPREAD_FLOOR_RATIO = 0.02

    def _apply_spread_floor(self, iqr: np.ndarray) -> np.ndarray:
        """
        Floor the per-channel error spread and record which channels were hit.

        Scaling by each channel's own median error keeps this comparable across
        datasets and channels carrying different units. Channels whose median
        error is also zero fall back to the typical channel, since they offer no
        scale of their own.
        """
        scale = np.asarray(self._err_median_, dtype=float).copy()
        positive = scale[scale > 0]
        scale[scale <= 0] = np.median(positive) if positive.size else 1.0

        floor = self._SPREAD_FLOOR_RATIO * scale
        self._degenerate_ = np.flatnonzero(iqr < floor)
        return np.maximum(iqr, floor)

    def degenerate_channels_(self) -> np.ndarray:
        """
        Channels whose forecast error had no measurable spread in training.

        Their spread was floored at fit time, because dividing by it would
        otherwise yield a number set by the guard epsilon rather than by the
        data. They can still raise alarms and still score highly when they move;
        the floor only stops them from dwarfing every other channel.

        Returns
        -------
        np.ndarray
            Indices into the feature axis, ascending.
        """
        self._require_fit()
        return np.asarray(self._degenerate_, dtype=int)

    def _warn_degenerate_spread(self) -> None:
        """
        Report channels whose spread was floored and that may raise an alarm.

        A record of what was done rather than advice: the floor is already
        applied. Only the scoring channels are named, since a floored channel
        excluded by ``score_channels`` cannot affect a score either way.
        """
        degenerate = set(self.degenerate_channels_().tolist())
        scoring = degenerate.intersection(
            range(self.n_nodes_) if self.score_channels is None else self.score_channels
        )
        if not scoring:
            return
        listed = ", ".join(str(c) for c in sorted(scoring))
        warnings.warn(
            f"Channels [{listed}] had no measurable spread in the training forecast "
            "error, so their spread was floored to keep their deviations bounded. "
            "This is usually a status or mode channel that stayed constant while "
            "training. They can still raise alarms, and will score highly the first "
            "time they move; exclude them with score_channels to keep them as "
            "context only.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _normalise(self, errors: np.ndarray) -> np.ndarray:
        """
        Per-node errors divided by their training spread, shape unchanged.

        Normalising by each node's own training median and IQR is what makes
        deviations comparable across channels carrying different units and
        dynamic ranges.
        """
        return np.abs(errors - self._err_median_) / (self._err_iqr_ + 1e-9)

    def _deviation_score(self, errors: np.ndarray) -> np.ndarray:
        """
        Collapse per-node errors to a single graph deviation score per window.

        Each node's error is normalised by its training median and IQR, then the
        maximum is taken over the scoring channels (the sensor deviating most
        drives the window score).

        Which channels contribute is set by ``score_channels``. Every channel
        still informs the graph and the forecast; this selects only which
        deviations are allowed to raise an alarm. See the constructor for why
        that distinction matters.
        """
        normed = self._normalise(errors)
        if self.score_channels is not None:
            normed = normed[:, self.score_channels]
        return normed.max(axis=1)

    def _errors_for(self, X: np.ndarray) -> np.ndarray:
        """
        Validate ``X`` against the fitted geometry and return per-node errors.

        Shared by every public scoring path so validation and preprocessing
        cannot drift between them.
        """
        self._require_fit()
        X = self._validate_X(X)
        if X.shape[2] != self.n_nodes_:
            raise ValueError(f"X has {X.shape[2]} features but GDN was fitted on {self.n_nodes_}.")
        if X.shape[1] - 1 != self.window_:
            raise ValueError(
                f"X has window_size {X.shape[1]} but GDN was fitted on "
                f"window_size {self.window_ + 1}."
            )
        X = self._scale_transform(X)
        return self._forecast_errors(X)

    def _build_net(self):
        """
        Construct the forecasting network. Subclasses override this to swap in a
        different architecture (e.g. KAN-GAT) while reusing the whole fit /
        scaling / deviation-scoring pipeline. Requires ``n_nodes_`` and
        ``window_`` to be set (done at the top of ``fit``).
        """
        from ._net import GDNNet

        return GDNNet(
            n_nodes=self.n_nodes_,
            window=self.window_,
            embed_dim=self.embed_dim,
            topk=self.topk,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> GDN:
        """
        Fit the GDN forecasting network on nominal telemetry windows.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)
            Windowed telemetry tensor from windowify(). ``window_size`` must be
            at least 2 (one context step plus one target step).
        y : ignored
            Present for API consistency.

        Returns
        -------
        self : GDN
        """
        torch = _import_torch()

        X = self._validate_X(X)
        if X.shape[1] < 2:
            raise ValueError(
                f"GDN needs window_size >= 2 (context + target), got window_size = {X.shape[1]}."
            )

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self.n_nodes_ = X.shape[2]
        self.window_ = X.shape[1] - 1
        device = self._resolve_device(torch)

        # Standardise inputs before the network sees them (fit on train only).
        X = self._scale_fit(X)

        context, target = self._split_context_target(X)
        ctx_t = torch.as_tensor(context, dtype=torch.float32)
        tgt_t = torch.as_tensor(target, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(ctx_t, tgt_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net = self._build_net().to(device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        self.net.train()
        for _ in range(self.epochs):
            for xb, tb in loader:
                xb = xb.to(device)
                tb = tb.to(device)
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, tb)
                loss.backward()
                optimizer.step()

        if self.score_channels is not None:
            beyond = [c for c in self.score_channels if c >= self.n_nodes_]
            if beyond:
                raise ValueError(
                    f"score_channels {beyond} exceed the {self.n_nodes_} fitted channels"
                )

        # Per-node training-error statistics for deviation normalisation.
        errors = self._forecast_errors(X)  # (n_windows, n_nodes)
        self._err_median_ = np.median(errors, axis=0)
        q75, q25 = np.percentile(errors, [75, 25], axis=0)
        self._err_iqr_ = self._apply_spread_floor(q75 - q25)

        scores = self._deviation_score(errors)
        self._set_post_fit(scores)
        # After _set_post_fit: the detector must read as fitted before the
        # public degenerate_channels_() accessor will answer.
        self._warn_degenerate_spread()
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the graph deviation score for each window.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)

        Returns
        -------
        scores : np.ndarray, shape (n_windows,)
            Graph deviation scores. Higher = more anomalous.
        """
        return self._deviation_score(self._errors_for(X))

    @property
    def scoring_channels_(self) -> list[int]:
        """
        Indices of the channels allowed to raise an alarm, in original order.

        Every channel is listed when ``score_channels`` is None.
        """
        self._require_fit()
        if self.score_channels is None:
            return list(range(self.n_nodes_))
        return list(self.score_channels)

    def learned_graph(self) -> dict:
        """
        The relational graph the detector learned over channels.

        Node embeddings are frozen after training and the graph is their top-k
        cosine similarity, so this is a property of the fitted model rather than
        of any particular input.

        Reading it is the only way to tell whether the graph found real
        structure. On data where each record holds one true sensor beside
        command flags, the neighbours are arbitrary and the graph contributes
        nothing, which metrics reveal only indirectly.

        Returns
        -------
        dict
            ``adjacency`` (bool, ``(n_nodes, n_nodes)``) where ``[i, j]`` is
            True when j is a neighbour of i; ``similarity`` (float, same shape)
            of pairwise cosine similarities with the diagonal set to ``-inf``;
            and ``embeddings`` (``(n_nodes, embed_dim)``).

        Notes
        -----
        The adjacency here carries no self-loops, which suits inspection and
        drawing. The distilled artifact's ``adj`` is the same matrix with the
        diagonal set, because attention there runs over ``topk + 1`` terms per
        node, the neighbours plus the node itself.
        """
        self._require_fit()
        from .distill import _topk_adjacency

        embeddings = self.net.encoder.embedding.weight.detach().cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.maximum(norms, 1e-12)
        similarity = normed @ normed.T
        np.fill_diagonal(similarity, -np.inf)
        return {
            "adjacency": _topk_adjacency(embeddings, self.topk),
            "similarity": similarity,
            "embeddings": embeddings,
        }

    def channel_deviations(self, X: np.ndarray) -> np.ndarray:
        """
        Per-channel normalised deviation for each window.

        This is the attribution behind :meth:`decision_function`, which reports
        only the maximum over the scoring channels. The full matrix answers
        *which* channels deviated and by how much, so an alarm can be explained
        rather than merely raised.

        The values are exact, not estimated. A perturbation method such as SHAP
        approximates a black box; here the per-channel deviation is what the
        detector already computes on its way to a score.

        Every channel is returned, including those excluded from
        ``score_channels``. A channel that may not raise an alarm can still be
        informative when diagnosing one, and callers that need only the scoring
        subset can index with :attr:`scoring_channels_`.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)

        Returns
        -------
        deviations : np.ndarray, shape (n_windows, n_features)
            Normalised absolute deviation per window and channel. Higher = the
            channel departed further from its forecast, in units of its own
            training spread.

        See Also
        --------
        dominant_channels : the channel driving each window's score.
        """
        return self._normalise(self._errors_for(X))

    def dominant_channels(self, X: np.ndarray) -> np.ndarray:
        """
        Index of the channel driving each window's score.

        Restricted to the scoring channels and returned in original channel
        indexing, so the result always names the channel whose deviation equals
        that window's :meth:`decision_function` score. Taking ``argmax`` over
        :meth:`channel_deviations` instead can name a context-only channel that
        never contributed to the score.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)

        Returns
        -------
        channels : np.ndarray, shape (n_windows,)
            Channel index per window, into the original feature axis.
        """
        normed = self._normalise(self._errors_for(X))
        scoring = self.scoring_channels_
        local = normed[:, scoring].argmax(axis=1)
        return np.asarray(scoring, dtype=int)[local]

    def _get_params(self) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "topk": self.topk,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "scale": self.scale,
            "percentile": self.percentile,
            "smoothing": self.smoothing,
            "score_channels": self.score_channels,
        }
