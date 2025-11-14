# src/telemetry_anomdet/models/unsupervised/pca.py

"""
PCA based anomaly detection.

This model learns a low dimensional subspace of nominal telemetry features
using Principal Component Analysis (PCA). Anomaly scores are computed as
the reconstruction error when projecting samples into the PCA subspace
and back into the original space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..base import BaseModel


@dataclass
class PCAAnomaly(BaseModel):
    """
    PCA based anomaly detection.

    The model fits a PCA transform on nominal telemetry feature vectors and
    uses reconstruction error as an anomaly score: points that cannot be
    well-expressed in the learned subspace have larger errors and are
    considered more anomalous.

    Attributes
    ----------
    n_components : int or None
        Number of principal components to keep. If None, all components are
        kept. In practice, you may choose this to retain a desired fraction
        of variance (via `explained_variance_ratio_` after fitting).
    scale : bool
        Whether to apply `StandardScaler` normalization before fitting PCA.
        Recommended when features differ significantly in scale.
    percentile : float
        Percentile of training reconstruction errors used to compute the
        default anomaly threshold in `is_anomaly()`. For example, 99.0 means
        that the top 1% highest-error training samples are considered
        anomalous by default.
    model : Optional[PCA]
        The fitted scikit-learn `PCA` instance after calling `fit()`.
        This is `None` before fitting.
    scaler : Optional[StandardScaler]
        The fitted `StandardScaler` when `scale = True`, otherwise `None`.
        Used to transform future inputs consistently with training.
    _train_scores : Optional[np.ndarray]
        Reconstruction error scores for the training set, computed after
        fitting. Used to derive the default anomaly threshold.
    _default_threshold : Optional[float]
        Default anomaly cutoff derived from training scores based on
        the configured `percentile`.
    """

    n_components: Optional[int] = None
    scale: bool = True
    percentile: float = 99.0

    # Fit artifacts
    model: Optional[PCA] = field(default = None, init = False)
    scaler: Optional[StandardScaler] = field(default=None, init = False)
    _train_scores: Optional[np.ndarray] = field(default = None, init = False)
    _default_threshold: Optional[float] = field(default = None, init = False)

    # ---- helpers ----
    def _require_fit(self):
        if self.model is None:
            raise RuntimeError("Model must be fitted before use. Call 'fit(X)' first.")

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(
                f"Input X must be 2D array (n_samples, n_features), got shape {X.shape}"
            )
        if X.shape[0] == 0:
            raise ValueError("Input X has 0 rows")
        if X.size == 0:
            raise ValueError("Input X has 0 elements")
        if not np.isfinite(X).all():
            raise ValueError("Input X contains NaN or infinite values")
        return X

    def _maybe_scale_fit(self, X: np.ndarray) -> np.ndarray:
        if self.scale:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        self.scaler = None
        return X

    def _maybe_scale_transform(self, X: np.ndarray) -> np.ndarray:
        if self.scale:
            if self.scaler is None:
                raise RuntimeError(
                    "Scaler not fitted. Was the model fitted with scale = True?"
                )
            return self.scaler.transform(X)
        return X

    # ---- core model interface ----
    def fit(self, X: np.ndarray):
        """
        Fit the PCA model to the telemetry feature matrix.

        Parameters
        ----------
        X : np.ndarray
            2D telemetry feature matrix of shape (n_samples, n_features).
            Each row corresponds to a time window or aggregated segment of telemetry.

        Returns
        -------
        self : PCAAnomaly
            Fitted model instance for chaining.
        """

        # TODO: validate X, fit PCA, compute training reconstruction errors, and set _train_scores/_default_threshold.

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores (reconstruction error) for each sample.

        Parameters
        ----------
        X : np.ndarray
            2D telemetry feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) containing anomaly scores, where higher
            values correspond to greater anomaly likelihood.
        """
        return self.score_samples(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error anomaly scores for each sample.

        Parameters
        ----------
        X : np.ndarray
            2D array of telemetry feature vectors.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) containing reconstruction-error scores.
            Higher values typically correspond to more anomalous points.
        """

        # TODO: implement reconstruction error using the fitted PCA model

        raise NotImplementedError

    def is_anomaly(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Flag samples as anomalies based on reconstruction error.

        Parameters
        ----------
        X : np.ndarray
            2D array of telemetry feature vectors to evaluate.
        threshold : float, optional
            Reconstruction-error cutoff for anomaly detection. Samples with
            scores greater than this value are considered anomalies. If None,
            the default threshold from the training data is used, based on
            `self.percentile`.

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_samples,) where True indicates an anomaly.
        """

        # TODO: call score_samples(), pick a threshold (either given or _default_threshold), and return a boolean mask

        raise NotImplementedError