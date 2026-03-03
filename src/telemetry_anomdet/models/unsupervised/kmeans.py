# src/telemetry_anomdet/models/unsupervised/kmeans.py

"""
K-Means clustering anomaly detection.

Each telemetry window is assigned to its nearest cluster centroid.
Anomaly scores are distances to the nearest centroid. Windows far from
any learned nominal operating mode score higher and are flagged as anomalies.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..base import BaseDetector
from ...feature_extraction import features

class KMeansAnomaly(BaseDetector):
    """
    K-Means clustering based anomaly detector.

    Accepts 3D windowed input (n_windows, window_size, n_features) and
    flattens internally via features_stat() before clustering. The caller
    never needs to manage this conversion.

    Parameters
    ----------
    n_clusters : int, default = 5
        Number of clusters (nominal operating modes) to learn. Each cluster
        represents a recurring pattern in the telemetry.
    scale : bool, default = False
        Apply StandardScaler before clustering. Enable when channels differ
        significantly in scale so distance calculations are not dominated by
        high-magnitude channels.
    percentile : float, default = 95.0
        Percentile of training centroid distances used to set the default
        anomaly threshold. 95.0 means the top 5% most distant training
        windows are labelled as anomalies.

    Attributes (set after fit)
    --------------------------
    decision_scores_ : np.ndarray, shape (n_windows,)
        Distance-to-nearest-centroid scores on training data.
    threshold_ : float
        Default anomaly cutoff derived from training scores at ``percentile``.
    labels_ : np.ndarray, shape (n_windows,)
        Binary anomaly labels on training data. 0 = normal, 1 = anomaly.
    model : sklearn.cluster.KMeans
        Fitted KMeans instance.
    centroids : np.ndarray, shape (n_clusters, n_features)
        Learned cluster centers in the (optionally scaled) feature space.
    scaler : sklearn.preprocessing.StandardScaler or None
        Fitted scaler when ``scale=True``, otherwise None.
    """

    def __init__(self, n_clusters: int = 5, scale: bool = False, percentile: float = 95.0,):
        super().__init__(percentile=percentile)
        self.n_clusters = n_clusters
        self.scale = scale

        # Fit artifacts: set in fit()
        self.model: Optional[KMeans] = None
        self.centroids: Optional[np.ndarray] = None
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
                    "Scaler is not fitted. Was the model fitted with scale=True?"
                )
            return self.scaler.transform(X2d)
        return X2d

    def _centroid_distances(self, Xs: np.ndarray) -> np.ndarray:
        """Distance from each sample to its nearest centroid."""
        return self.model.transform(Xs).min(axis = 1)

    # ---- BaseDetector interface ----
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KMeansAnomaly":
        """
        Fit K-Means on nominal telemetry windows.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)
            Windowed telemetry tensor from windowify().
        y : ignored
            Present for API consistency.

        Returns
        -------
        self : KMeansAnomaly

        Raises
        ------
        ValueError
            If n_clusters < 1 or n_clusters > n_windows.
        """
        X = self._validate_X(X)
        X2d = self._flatten(X)
        n_windows = X2d.shape[0]

        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if self.n_clusters > n_windows:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot exceed the number of "
                f"windows ({n_windows}). Reduce n_clusters or increase your data."
            )

        Xs = self._scale_fit(X2d)

        self.model = KMeans(
            n_clusters = self.n_clusters,
            n_init = 10,
            random_state = 0,
        )
        self.model.fit(Xs)
        self.centroids = self.model.cluster_centers_

        scores = self._centroid_distances(Xs)
        self._set_post_fit(scores)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute distance-to-nearest-centroid scores for each window.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)

        Returns
        -------
        scores : np.ndarray, shape (n_windows,)
            Centroid distances. Higher = more anomalous.
        """
        self._require_fit()
        X = self._validate_X(X)
        X2d = self._flatten(X)
        Xs = self._scale_transform(X2d)
        return self._centroid_distances(Xs)

    # ---- KMeans specific method ----
    def predict_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each window to its nearest cluster.

        Unique to KMeansAnomaly. Not part of the BaseDetector interface.
        Useful for understanding which nominal operating mode each window
        belongs to, independent of whether it is flagged as an anomaly.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)

        Returns
        -------
        cluster_labels : np.ndarray of int, shape (n_windows,)
            Cluster index in [0, n_clusters - 1] for each window.
        """
        self._require_fit()
        X = self._validate_X(X)
        X2d = self._flatten(X)
        Xs = self._scale_transform(X2d)
        return self.model.predict(Xs)

    # ---- Repr support ----
    def _get_params(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "scale": self.scale,
            "percentile": self.percentile,
        }