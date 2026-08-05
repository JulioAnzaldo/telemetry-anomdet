# src/telemetry_anomdet/models/deep/distill.py

"""
Symbolic distillation of a fitted KANGDN into a torch-free evaluator.

A fitted :class:`~telemetry_anomdet.models.deep._kan.KANLayer` is already a
closed-form function: every edge is ``phi_ij(x) = w_base * silu(x) + sum_k
theta_ijk * B_k(x)`` (a SiLU term plus a B-spline, both exact). This module pulls
those learned coefficients out of the torch modules into plain dicts and rebuilds
the forward pass in **pure NumPy**, no torch at inference.

The NumPy evaluator is the portable, dependency-free form of the detector, and is
the intended source for a C port onto a microcontroller. The equivalence tests
(NumPy output == torch output) are the guarantee that the distilled artifact
behaves exactly like the trained model.

Two levels are provided:

per layer
    :func:`extract_kan_layer` / :class:`KANLayerNumpy`, one KAN layer, plus
    :meth:`KANLayerNumpy.edge_function` for the atomic 1-D edge ``phi_ij(x)``
    that later feeds symbolic regression.

whole detector
    :func:`extract_kan_gdn` / :class:`KANGDNNumpy`, the entire scoring path of a
    fitted ``KANGDN``: per-channel scaler, GAT encoder (frozen graph, learned
    attention), both KAN layers, the deviation normalisation and the threshold.
    This is the deployable artifact, containing everything needed to turn a
    window of telemetry into an anomaly flag.

Two things collapse at extraction time and never need recomputing on device:

* **The graph is frozen.** ``topk_graph`` depends only on the learned embeddings,
  so the top-k adjacency is resolved once here and stored as a boolean matrix.
  No cosine similarities or top-k sorts at inference.
* **Attention factorises.** The scores are ``a^T [g_i | g_j]``, and ``a`` splits
  into halves, so ``score_ij = a_src . g_i + a_dst . g_j`` is an outer sum of two
  length-``n_nodes`` vectors. The torch code materialises an
  ``(n, n, 4 * embed_dim)`` tensor; the distilled form never does.

This module imports only NumPy; it reads fitted torch modules via their tensors
but never imports torch, so the distilled path is torch-free.
"""

from __future__ import annotations

import numpy as np


def extract_kan_layer(layer) -> dict:
    """
    Pull a fitted ``KANLayer``'s learned parameters into a torch-free dict.

    Parameters
    ----------
    layer : KANLayer
        A fitted KAN layer (its ``grid``, ``base_weight``, ``spline_weight`` are
        read and detached to NumPy).

    Returns
    -------
    dict
        ``in_features``, ``out_features``, ``spline_order`` (ints) and ``grid``,
        ``base_weight``, ``spline_weight`` (NumPy arrays). Fully describes the
        layer's function; can be JSON/NPZ-serialised for deployment.
    """
    return {
        "in_features": int(layer.in_features),
        "out_features": int(layer.out_features),
        "spline_order": int(layer.spline_order),
        "grid": layer.grid.detach().cpu().numpy().astype(float),
        "base_weight": layer.base_weight.detach().cpu().numpy().astype(float),
        "spline_weight": layer.spline_weight.detach().cpu().numpy().astype(float),
    }


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU / swish activation: x * sigmoid(x)."""
    return x / (1.0 + np.exp(-x))


def _b_splines_np(x: np.ndarray, grid: np.ndarray, spline_order: int) -> np.ndarray:
    """
    NumPy port of ``KANLayer.b_splines`` (Cox-de Boor recursion).

    Parameters
    ----------
    x : np.ndarray, shape (n, in_features)
    grid : np.ndarray, shape (in_features, grid_size + 2*spline_order + 1)
    spline_order : int

    Returns
    -------
    bases : np.ndarray, shape (n, in_features, grid_size + spline_order)
    """
    x = x[..., np.newaxis]  # (n, in_features, 1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)
    for k in range(1, spline_order + 1):
        left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
        right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k])
        bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
    return bases


class KANLayerNumpy:
    """
    Torch-free evaluator of a distilled KAN layer.

    Reconstructed from :func:`extract_kan_layer`, this reproduces the torch
    ``KANLayer`` forward pass using only NumPy, giving the portable form suitable
    for a C port. ``forward`` accepts ``(..., in_features)`` and returns
    ``(..., out_features)``.
    """

    def __init__(self, extracted: dict):
        self.in_features = extracted["in_features"]
        self.out_features = extracted["out_features"]
        self.spline_order = extracted["spline_order"]
        self.grid = np.asarray(extracted["grid"], dtype=float)
        self.base_weight = np.asarray(extracted["base_weight"], dtype=float)
        self.spline_weight = np.asarray(extracted["spline_weight"], dtype=float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lead_shape = x.shape[:-1]
        x = x.reshape(-1, self.in_features)  # (n, in_features)

        base = _silu(x) @ self.base_weight.T  # (n, out_features)

        bases = _b_splines_np(x, self.grid, self.spline_order)  # (n, in, coeff)
        spline = bases.reshape(x.shape[0], -1) @ self.spline_weight.reshape(self.out_features, -1).T

        y = base + spline
        return y.reshape(*lead_shape, self.out_features)

    __call__ = forward

    def edge_function(self, out_idx: int, in_idx: int):
        """
        Return the single 1-D edge function ``phi_ij(x)`` as a NumPy callable.

        This is the atomic unit of KAN distillation: each edge is a function of
        one input, ready to sample, plot, or hand to symbolic regression.

        Parameters
        ----------
        out_idx, in_idx : int
            Output and input indices selecting the edge.

        Returns
        -------
        callable
            ``phi(x)`` for scalar or array ``x``, where the layer output
            ``y[out_idx] = sum_in phi(x[in_idx])``.
        """
        grid_i = self.grid[in_idx : in_idx + 1]  # (1, knots)
        w_base = self.base_weight[out_idx, in_idx]
        coeffs = self.spline_weight[out_idx, in_idx]  # (grid_size + spline_order,)

        def phi(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float).reshape(-1, 1)  # (n, 1) as one feature
            base = w_base * _silu(x[:, 0])
            bases = _b_splines_np(x, grid_i, self.spline_order)[:, 0, :]  # (n, coeff)
            return base + bases @ coeffs

        return phi


# ---------------------------------------------------------------------------
# Whole-network distillation: GATEncoder + both KAN layers
# ---------------------------------------------------------------------------


def _topk_adjacency(embeddings: np.ndarray, k: int) -> np.ndarray:
    """
    NumPy port of ``_net.topk_graph``, resolved once at extraction time.

    The graph depends only on the learned embeddings, which are frozen after
    training, so the distilled artifact carries the adjacency as data rather
    than recomputing cosine similarities and a top-k sort on device.

    Ties are broken by lower node index (stable sort on the negated similarity),
    matching ``torch.topk``.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_nodes, embed_dim)
    k : int
        Neighbours per node, clamped to ``n_nodes - 1``.

    Returns
    -------
    adj : np.ndarray of bool, shape (n_nodes, n_nodes)
        ``adj[i, j]`` is True when j is a neighbour of i.
    """
    n_nodes = embeddings.shape[0]
    k = max(1, min(k, n_nodes - 1))

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # torch's normalize clamps the denominator rather than dividing by zero.
    normed = embeddings / np.maximum(norms, 1e-12)
    sim = normed @ normed.T
    np.fill_diagonal(sim, -np.inf)  # never select self

    topk_idx = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    adj = np.zeros((n_nodes, n_nodes), dtype=bool)
    np.put_along_axis(adj, topk_idx, True, axis=1)
    return adj


def extract_kan_gdn_net(net) -> dict:
    """
    Pull a fitted ``KANGDNNet``'s learned parameters into a torch-free dict.

    Resolves the learned top-k graph and splits the attention vector into its
    source and target halves, so the distilled form has no graph construction
    and no ``(n, n, 4 * embed_dim)`` intermediate left to compute.

    Parameters
    ----------
    net : KANGDNNet
        A fitted KAN-GAT forecasting network.

    Returns
    -------
    dict
        ``n_nodes``, ``window``, ``embed_dim`` (ints); ``embedding``,
        ``feat_weight``, ``feat_bias``, ``attn_src``, ``attn_dst``, ``adj``
        (NumPy arrays); ``leaky_slope`` (float); and ``activation`` / ``out``,
        the two extracted KAN layers.
    """
    encoder = net.encoder
    embedding = encoder.embedding.weight.detach().cpu().numpy().astype(float)
    attn = encoder.attn.weight.detach().cpu().numpy().astype(float).reshape(-1)
    half = attn.shape[0] // 2  # 2 * embed_dim

    adj = _topk_adjacency(embedding, int(encoder.topk))
    np.fill_diagonal(adj, True)  # self-loops, as in the torch forward

    return {
        "n_nodes": int(encoder.n_nodes),
        "window": int(encoder.window),
        "embed_dim": int(encoder.embed_dim),
        "embedding": embedding,
        "feat_weight": encoder.feat.weight.detach().cpu().numpy().astype(float),
        "feat_bias": encoder.feat.bias.detach().cpu().numpy().astype(float),
        # score_ij = a_src . g_i + a_dst . g_j
        "attn_src": attn[:half],
        "attn_dst": attn[half:],
        "leaky_slope": float(encoder.leaky.negative_slope),
        "adj": adj,
        "activation": extract_kan_layer(encoder.activation),
        "out": extract_kan_layer(net.out),
    }


def _masked_softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis, with non-neighbours zeroed out."""
    scores = np.where(mask, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)  # self-loop keeps a finite max
    exp = np.exp(scores)
    return exp / exp.sum(axis=-1, keepdims=True)


class KANGDNNetNumpy:
    """
    Torch-free evaluator of a distilled ``KANGDNNet``.

    Reconstructed from :func:`extract_kan_gdn_net`, this reproduces the whole
    forecasting network (graph attention encoder, KAN activation, embedding gate,
    KAN forecast head) using only NumPy.

    ``forward`` accepts ``(batch, n_nodes, window)`` and returns the one-step
    forecast ``(batch, n_nodes)``, matching ``KANGDNNet.forward``.
    """

    def __init__(self, extracted: dict):
        self.n_nodes = extracted["n_nodes"]
        self.window = extracted["window"]
        self.embed_dim = extracted["embed_dim"]
        self.embedding = np.asarray(extracted["embedding"], dtype=float)
        self.feat_weight = np.asarray(extracted["feat_weight"], dtype=float)
        self.feat_bias = np.asarray(extracted["feat_bias"], dtype=float)
        self.attn_src = np.asarray(extracted["attn_src"], dtype=float)
        self.attn_dst = np.asarray(extracted["attn_dst"], dtype=float)
        self.leaky_slope = float(extracted["leaky_slope"])
        self.adj = np.asarray(extracted["adj"], dtype=bool)
        self.activation = KANLayerNumpy(extracted["activation"])
        self.out = KANLayerNumpy(extracted["out"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        v = self.embedding  # (n_nodes, embed_dim)
        h = x @ self.feat_weight.T + self.feat_bias  # (batch, n_nodes, embed_dim)

        # Node descriptor g_i = [v_i | W x_i], broadcast across the batch.
        vb = np.broadcast_to(v, h.shape)
        g = np.concatenate([vb, h], axis=-1)  # (batch, n_nodes, 2*embed_dim)

        # pi(i, j) = LeakyReLU(a_src . g_i + a_dst . g_j): an outer sum, not a
        # pairwise concatenation.
        src = g @ self.attn_src  # (batch, n_nodes)
        dst = g @ self.attn_dst  # (batch, n_nodes)
        scores = src[:, :, np.newaxis] + dst[:, np.newaxis, :]
        scores = np.where(scores >= 0, scores, self.leaky_slope * scores)

        weights = _masked_softmax(scores, self.adj[np.newaxis])

        # z_i = activation(sum_j alpha_ij W x_j) * v_i
        z = weights @ h  # (batch, n_nodes, embed_dim)
        z = self.activation(z)
        z = z * v
        return self.out(z)[..., 0]  # (batch, n_nodes)

    __call__ = forward


# ---------------------------------------------------------------------------
# Whole-detector distillation: scaler + network + deviation scoring + threshold
# ---------------------------------------------------------------------------


def extract_kan_gdn(detector) -> dict:
    """
    Distil a fitted :class:`~telemetry_anomdet.models.deep.kan_gdn.KANGDN` into a
    torch-free specification of its complete scoring path.

    Parameters
    ----------
    detector : KANGDN
        A fitted detector (``fit`` must have been called).

    Returns
    -------
    dict
        ``net`` (see :func:`extract_kan_gdn_net`); ``scaler_mean`` /
        ``scaler_scale`` (NumPy arrays, or ``None`` when ``scale=False``);
        ``err_median`` / ``err_iqr`` (per-node deviation normalisation);
        ``threshold`` (float). Every array is NumPy, so the whole dict is
        NPZ-serialisable and is what the C export will read.

    Raises
    ------
    RuntimeError
        If the detector has not been fitted.
    """
    if getattr(detector, "net", None) is None:
        raise RuntimeError("Detector is not fitted. Call fit() before distilling.")

    scaler = getattr(detector, "scaler", None)
    return {
        "net": extract_kan_gdn_net(detector.net),
        "scaler_mean": None if scaler is None else np.asarray(scaler.mean_, dtype=float),
        "scaler_scale": None if scaler is None else np.asarray(scaler.scale_, dtype=float),
        "err_median": np.asarray(detector._err_median_, dtype=float),
        "err_iqr": np.asarray(detector._err_iqr_, dtype=float),
        "threshold": float(detector.threshold_),
    }


class KANGDNNumpy:
    """
    Torch-free evaluator of a distilled ``KANGDN`` detector.

    The deployable artifact: takes raw telemetry windows and returns deviation
    scores and anomaly flags identical to the fitted torch detector, using only
    NumPy. Mirrors the detector's ``decision_function`` / ``predict``.

    Parameters
    ----------
    extracted : dict
        Output of :func:`extract_kan_gdn`.

    Notes
    -----
    Input is ``(n_windows, window_size, n_features)``, the same 3-D window tensor
    the detector is fitted on. Scaling, the context/target split, the forecast,
    the per-node deviation normalisation and the threshold comparison all happen
    here, so nothing outside this class is needed at inference.
    """

    def __init__(self, extracted: dict):
        self.net = KANGDNNetNumpy(extracted["net"])
        mean, scale = extracted["scaler_mean"], extracted["scaler_scale"]
        self.scaler_mean = None if mean is None else np.asarray(mean, dtype=float)
        self.scaler_scale = None if scale is None else np.asarray(scale, dtype=float)
        self.err_median = np.asarray(extracted["err_median"], dtype=float)
        self.err_iqr = np.asarray(extracted["err_iqr"], dtype=float)
        self.threshold = float(extracted["threshold"])

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """Apply the per-channel standardiser (identity when unscaled)."""
        if self.scaler_mean is None:
            return X
        return (X - self.scaler_mean) / self.scaler_scale

    def forecast_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Per-window, per-node absolute forecast error, shape ``(n_windows, n_nodes)``.
        """
        X = self._scale(np.asarray(X, dtype=float))
        context = np.transpose(X[:, :-1, :], (0, 2, 1))  # (n, n_nodes, window)
        target = X[:, -1, :]  # (n, n_nodes)
        return np.abs(self.net.forward(context) - target)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Graph deviation score per window (higher = more anomalous).

        Each node's forecast error is normalised by its training median and IQR;
        the score is the maximum across nodes, so the sensor deviating most
        drives the window.
        """
        errors = self.forecast_errors(X)
        normed = np.abs(errors - self.err_median) / (self.err_iqr + 1e-9)
        return normed.max(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary anomaly labels (1 = anomaly) from the distilled threshold."""
        return (self.decision_function(X) > self.threshold).astype(int)
