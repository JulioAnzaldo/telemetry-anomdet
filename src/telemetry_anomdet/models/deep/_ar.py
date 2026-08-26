# src/telemetry_anomdet/models/deep/_ar.py

"""
Fixed autoregressive memory filters (the "AR" of AR-KAN).

Wu et al. (2025), "AR-KAN: Autoregressive-Weight-Enabled Kolmogorov-Arnold
Network for Time Series Forecasting" (arXiv:2509.02967), factor a forecaster into
two stages justified by the Universal Myopic Mapping theorem: any myopic,
shift-invariant system is a bank of linear filters followed by a static
nonlinearity. The linear stage is an ordinary AR model whose coefficients are
solved in closed form by Yule-Walker and then **frozen**; the nonlinear stage is
a KAN over the filtered lags::

    y_i(n)   = a_i * x(n - i)                    i = 1 .. p     (fixed)
    x(n + 1) = KAN(y_1(n), ..., y_p(n))                         (learned)

The filter is diagonal: an element-wise scaling of the window, not a matrix. It
adds ``p`` frozen floats per channel, no trainable parameters, and no backward
pass. That is what makes it cheap enough to fly.

This module is deliberately torch-free NumPy so the coefficients can be computed,
inspected, and emitted to C without the deep extra installed.
"""

from __future__ import annotations

import numpy as np

#: Node feature transforms selectable by ``KANGDN(feat_mode=...)``.
#:
#: ``linear``
#:     ``nn.Linear(window, embed_dim)``. The default, and what both GDN and
#:     KANGDN have always used: raw telemetry enters the network through a
#:     single linear map and the splines only ever see the compressed embedding.
#: ``kan``
#:     A KAN over the raw window. Adds a nonlinearity on each node's own history
#:     without discarding any of it.
#: ``ar_kan``
#:     AR-KAN as published: a frozen Yule-Walker filter, then a KAN.
#: ``ar_kan_residual``
#:     ``Linear(x) + KAN(a * x)``. The AR-KAN branch corrects a linear baseline
#:     rather than replacing it, so the full window always reaches the encoder.
#:
#: ``Linear(x) + KAN(x)`` was offered as ``kan_residual`` and removed for never
#: being best in any regime measured; see :doc:`/user_guide/feature_transforms`.
#: ``ARKANFeatures`` still accepts the flag combination, since ``ar`` and
#: ``residual`` are independent, it is simply not offered as a mode.
#:
#: Lives here rather than in ``_kan`` so ``KANGDN.__init__`` can validate the
#: argument without importing torch.
FEAT_MODES = ("linear", "kan", "ar_kan", "ar_kan_residual")

# Diagonal loading on the Toeplitz system, as a fraction of the lag-0
# autocovariance. Near-deterministic channels give an almost singular R; without
# this the solve returns coefficients of order 1e8 that saturate the KAN grid.
_RIDGE = 1e-6

# Below this ratio of variance to mean-square a channel carries no usable
# autocorrelation and falls back to persistence. Matches the spirit of the
# spread floor in gdn.py: constant SMAP command flags must not blow up.
_MIN_VARIANCE = 1e-12


def _autocovariance(series: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Biased sample autocovariance, averaged over independent segments.

    Parameters
    ----------
    series : np.ndarray, shape (n_segments, length)
        Contiguous segments of one channel. Windows from ``make_feature_table``
        overlap, so they are not independent draws, but averaging their
        autocovariances is still a consistent estimator of the channel's.
    max_lag : int
        Highest lag to estimate. Lags approaching ``length`` rest on very few
        pairs.

    Returns
    -------
    acov : np.ndarray, shape (max_lag + 1,)

    Notes
    -----
    The **biased** normaliser (divide by ``length``, not ``length - k``) is not a
    sloppy choice: it is what guarantees the resulting Toeplitz matrix is
    positive semi-definite, so the Yule-Walker solve is well posed. It also
    shrinks the high lags toward zero, which is the regularisation those thin
    estimates need.

    Centring uses the **global** mean, not each segment's own. Segments are
    short slices of one channel, and a persistent process is exactly what makes
    their individual means wander; removing those means would subtract the
    low-frequency content the AR fit is there to capture. On an AR(1) with
    phi = 0.9 in windows of 32, per-segment centring recovers 0.64.
    """
    n_segments, length = series.shape
    centred = series - series.mean()

    acov = np.zeros(max_lag + 1, dtype=float)
    denom = float(n_segments * length)
    for lag in range(min(max_lag, length - 1) + 1):
        acov[lag] = float((centred[:, : length - lag] * centred[:, lag:]).sum()) / denom
    return acov


def yule_walker(series: np.ndarray, order: int) -> np.ndarray:
    """
    Solve the Yule-Walker equations for the AR coefficients of one channel.

    Fits ``x(n) = sum_{i=1..p} a_i x(n - i)`` by matching the sample
    autocovariance: ``R a = r`` with ``R[i, j] = acov(|i - j|)`` and
    ``r[i] = acov(i + 1)``.

    Parameters
    ----------
    series : np.ndarray, shape (n_segments, length)
        Contiguous segments of the channel.
    order : int
        AR order ``p``.

    Returns
    -------
    coef : np.ndarray, shape (order,)
        ``coef[i]`` multiplies lag ``i + 1``. A channel with no usable variance
        returns persistence, ``[1, 0, ..., 0]``, rather than a singular solve.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    persistence = np.zeros(order, dtype=float)
    persistence[0] = 1.0

    acov = _autocovariance(np.atleast_2d(series).astype(float), order)
    if acov[0] <= _MIN_VARIANCE:
        return persistence

    idx = np.arange(order)
    R = acov[np.abs(idx[:, None] - idx[None, :])]
    R = R + _RIDGE * acov[0] * np.eye(order)
    r = acov[1 : order + 1]

    try:
        coef = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        return persistence
    if not np.all(np.isfinite(coef)):
        return persistence
    return coef


def ar_filters(context: np.ndarray, order: int | None = None) -> np.ndarray:
    """
    Per-channel AR memory filters, aligned to the network's context layout.

    Parameters
    ----------
    context : np.ndarray, shape (n_windows, n_nodes, window)
        Per-node input sequences as the encoder sees them: the last axis runs
        oldest to newest.
    order : int or None, default=None
        AR order. Defaults to ``window``, giving one coefficient per timestep so
        the filter is an element-wise scaling of the whole context. A shorter
        order zeroes the oldest lags.

    Returns
    -------
    filters : np.ndarray, shape (n_nodes, window)
        Ready to multiply ``context`` element-wise. **Reversed relative to
        ``yule_walker``**: the encoder's last timestep is lag 1, so coefficient
        ``a_1`` lands at index ``window - 1``.
    """
    context = np.asarray(context, dtype=float)
    if context.ndim != 3:
        raise ValueError(f"context must be 3D (n_windows, n_nodes, window), got {context.shape}")

    _, n_nodes, window = context.shape
    order = window if order is None else int(order)
    if not 1 <= order <= window:
        raise ValueError(f"order must lie in [1, {window}], got {order}")

    filters = np.zeros((n_nodes, window), dtype=float)
    for node in range(n_nodes):
        coef = yule_walker(context[:, node, :], order)
        # coef[i] multiplies lag i+1; the context's newest sample is lag 1.
        filters[node, window - order :] = coef[::-1]
    return filters
