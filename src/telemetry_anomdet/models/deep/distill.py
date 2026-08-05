# src/telemetry_anomdet/models/deep/distill.py

"""
Symbolic distillation of KAN layers.

Step 1: torch-free extraction.

A fitted :class:`~telemetry_anomdet.models.deep._kan.KANLayer` is already a
closed-form function: every edge is ``phi_ij(x) = w_base * silu(x) + sum_k
theta_ijk * B_k(x)`` (a SiLU term plus a B-spline, both exact). This module pulls
those learned coefficients out of the torch module into a plain dict and rebuilds
the forward pass in **pure NumPy**, no torch at inference.

Why this matters for the application: the NumPy evaluator is the portable,
dependency-free form of the detector, the thing you port to C for an STM32. The
equivalence test (NumPy output == torch output) is the guarantee that the
distilled artifact behaves exactly like the trained model. Later steps sit on top
of this: per-edge symbolic regression (recognise ``sin``/polynomials for
human-auditable equations) and C code generation.

This module imports only NumPy; it reads a fitted torch ``KANLayer`` via its
tensors but never imports torch, so the distilled path is torch-free.
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
    ``KANLayer`` forward pass using only NumPy -- the portable form you would
    port to C. ``forward`` accepts ``(..., in_features)`` and returns
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
