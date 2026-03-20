# src/telemetry_anomdet/models/unsupervised/pca.py

"""
PCA based anomaly detection.

This model learns a low dimensional subspace of nominal telemetry features
using Principal Component Analysis (PCA). Anomaly scores are computed as
the reconstruction error when projecting samples into the PCA subspace
and back into the original space.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..base import BaseDetector
from ...feature_extraction import features

class PCAAnomaly(BaseDetector):
    """
    PCA-based anomaly detector.

    Accepts 3D windowed input (n_windows, window_size, n_features) and
    flattens internally via features_stat() before fitting PCA. The caller
    never needs to manage this conversion.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of principal components to retain. If None, all components
        are kept. Choose a value that retains your desired fraction of
        variance — inspect ``model.explained_variance_ratio_`` after fitting.
    scale : bool, default=True
        Apply StandardScaler before PCA. Recommended when telemetry channels
        differ significantly in scale (e.g. voltage vs. temperature).
    percentile : float, default = 95.0
        Percentile of training reconstruction errors used to set the default
        anomaly threshold. 95.0 means the top 5% most anomalous training
        windows are labelled as anomalies.

    Attributes (set after fit)
    --------------------------
    decision_scores_ : np.ndarray, shape (n_windows,)
        Reconstruction errors on training data.
    threshold_ : float
        Default anomaly cutoff derived from training scores at ``percentile``.
    labels_ : np.ndarray, shape (n_windows,)
        Binary anomaly labels on training data. 0 = normal, 1 = anomaly.
    model : sklearn.decomposition.PCA
        Fitted PCA instance.
    scaler : sklearn.preprocessing.StandardScaler or None
        Fitted scaler when ``scale = True``, otherwise None.
    """

    def __init__(self, n_components: Optional[int] = None, scale: bool = True, percentile: float = 95.0,):
        super().__init__(percentile = percentile)
        self.n_components = n_components
        self.scale = scale

        # Fit artifacts: set in fit()
        self.model: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None

    # ---- helpers ----
    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten 3D windowed tensor to 2D feature matrix via features_stat().

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)

        Returns
        -------
        X2d : np.ndarray, shape (n_windows, n_features * 6)
            Statistical features per window: mean, std, min, max, median, slope.
        """
        return features.features_stat(X)

    def _scale_fit(self, X2d: np.ndarray) -> np.ndarray:
        if self.scale:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X2d)
        self.scaler = None
        return X2d
    
    def _scale_transform(self, X2d: np.ndarray) -> np.ndarray:
        if self.scale:
            if self.scaler is None:
                raise RuntimeError(
                    "Scaler is not fitted. Was the model fitted with scale = True?"
                )
            return self.scaler.transform(X2d)
        return X2d

    def _reconstruction_error(self, Xs: np.ndarray) -> np.ndarray:
        """Project into PCA subspace and compute per-window reconstruction error."""
        Z = self.model.transform(Xs)
        X_recon = self.model.inverse_transform(Z)
        return np.sum((Xs - X_recon) ** 2, axis = 1)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PCAAnomaly":
        """
        Fit PCA on nominal telemetry windows.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)
            Windowed telemetry tensor from windowify().
        y : ignored
            Present for API consistency.

        Returns
        -------
        self : PCAAnomaly
        """
        X = self._validate_X(X)
        X2d = self._flatten(X)
        Xs = self._scale_fit(X2d)

        self.model = PCA(n_components = self.n_components)
        self.model.fit(Xs)

        scores = self._reconstruction_error(Xs)
        self._set_post_fit(scores)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for each window.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)

        Returns
        -------
        scores : np.ndarray, shape (n_windows,)
            Reconstruction errors. Higher = more anomalous.
        """
        self._require_fit()
        X = self._validate_X(X)
        X2d = self._flatten(X)
        Xs = self._scale_transform(X2d)
        return self._reconstruction_error(Xs)
    
    def _get_params(self) -> dict:
        return {
            "n_components": self.n_components,
            "scale": self.scale,
            "percentile": self.percentile,
        }