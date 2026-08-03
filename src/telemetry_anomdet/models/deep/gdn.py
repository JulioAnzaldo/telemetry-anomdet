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

        # Fit artifacts: set in fit()
        self.net = None
        self.scaler: StandardScaler | None = None
        self.n_nodes_: int | None = None
        self.window_: int | None = None
        self._err_median_: np.ndarray | None = None
        self._err_iqr_: np.ndarray | None = None

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
        return np.concatenate(errors, axis=0)

    def _deviation_score(self, errors: np.ndarray) -> np.ndarray:
        """
        Collapse per-node errors to a single graph deviation score per window.

        Each node's error is normalised by its training median and IQR, then the
        maximum across nodes is taken (the sensor deviating most drives the
        window score).
        """
        normed = np.abs(errors - self._err_median_) / (self._err_iqr_ + 1e-9)
        return normed.max(axis=1)

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
        from ._net import GDNNet

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

        self.net = GDNNet(
            n_nodes=self.n_nodes_,
            window=self.window_,
            embed_dim=self.embed_dim,
            topk=self.topk,
        ).to(device)

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

        # Per-node training-error statistics for deviation normalisation.
        errors = self._forecast_errors(X)  # (n_windows, n_nodes)
        self._err_median_ = np.median(errors, axis=0)
        q75, q25 = np.percentile(errors, [75, 25], axis=0)
        self._err_iqr_ = q75 - q25

        scores = self._deviation_score(errors)
        self._set_post_fit(scores)
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
        errors = self._forecast_errors(X)
        return self._deviation_score(errors)

    def _get_params(self) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "topk": self.topk,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "scale": self.scale,
            "percentile": self.percentile,
        }
