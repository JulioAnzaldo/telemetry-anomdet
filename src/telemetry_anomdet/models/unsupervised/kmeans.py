# src/telemetry_anomdet/models/unsupervised/kmeans.py

"""
K-Means Clustering for Anomaly Detection

Each telemetry window (feature vector) is assigned to the nearest cluster centroid.
Anomalies are identified based on their distance to the closest centroid.
"""

from __future__ import annotations

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from ..base import BaseModel
import numpy as np
from typing import Optional

@dataclass
class KMeansAnomaly(BaseModel):
    """
    K-Means clustering-based anomaly detection.

    Attributes
    ----------
    n_clusters : int
        Number of clusters (nominal operating modes) to identify.
    scale : bool
        Whether to apply `StandardScaler` normalization before clustering.
        Recommended when features differ in scale.
    percentile : float
        Percentile of training-set distances used to compute the default
        anomaly threshold in `is_anomaly()`. For example, 95.0 means that
        the top 5% most distant training samples are considered anomalous.
    model : Optional[KMeans]
        The fitted scikit-learn `KMeans` instance after calling `fit()`.
        This is `None` before fitting.
    scaler : Optional[StandardScaler]
        The fitted `StandardScaler` instance when `scale = True`, otherwise
        `None`. Used to transform future inputs consistently with training.
    centroids : Optional[np.ndarray]
        Array of shape `(n_clusters, n_features)` containing the learned
        cluster centers after fitting.

    Notes
    -----
    - This class follows the `BaseModel` API used across telemetry_anomdet.
      All models expose `fit()` and `predict()` for interoperability.
    - Although clustering is unsupervised, it can provide interpretable
      anomaly scores via the `score_samples()` method.
    - The anomaly threshold in `is_anomaly()` can be set manually or
      determined statistically (via percentiles of training distances).
    """

    n_clusters: int = 5
    scale: bool = False
    percentile: float = 95.0

    # Fit artifacts
    model: Optional[KMeans] = field(default = None, init = False)
    scaler: Optional[StandardScaler] = field(default = None, init = False)
    centroids: Optional[np.ndarray] = field(default = None, init = False)
    _train_distances: Optional[np.ndarray] = field(default = None, init = False)
    _default_threshold: Optional[float] = field(default = None, init = False)

    # ---- helpers ----
    def _require_fit(self):
        if self.model is None:
            raise RuntimeError("Model must be fitted before use. Call 'fit(X)' first pls.")

    def _validate_X(self, X: np.ndarray, *, allow_empty: bool = False) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Input X must be 2D array (n_samples, n_features), got shape {X.shape}")
        if not allow_empty and X.shape[0] == 0:
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
                raise RuntimeError("Scaler not fitted. Was model fitted with scale=True?.")
            return self.scaler.transform(X)
        return X

    # Core Model Interface
    def fit(self, X: np.ndarray):
        """
        Fit the K-Means model to the telemetry feature matrix.

        Parameters
        ----------
        X : np.ndarray
            2D telemetry feature matrix of shape (n_samples, n_features).
            Each row corresponds to a time window or aggregated segment of telemetry.

        Returns
        -------
        self : KMeansAnomaly
            Fitted model instance for chaining.
        """

        X = self._validate_X(X)
        n_samples = X.shape[0]

        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if self.n_clusters > n_samples:
            raise ValueError(f"n_clusters ({self.n_clusters}) cannot exceed number of samples ({n_samples})")

        Xs = self._maybe_scale_fit(X)


        # Use stable settings for reproducibility
        self.model = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=0,
        )
        self.model.fit(Xs)
        self.centroids = self.model.cluster_centers_

        #Distances of training points to their nearest centroid
        dists = self.model.transform(Xs).min(axis=1)
        self._train_distances = dists

        #Default threshold based on training distances
        p = float(self.percentile)
        if not (0.0 < p < 100.0):
            raise ValueError("percentile must be in (0, 100)")
        self._default_threshold = np.percentile(dists, p)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores for each sample.

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
    
    def predict_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample to its nearest cluster based on the fitted K-Means model.

        Parameters
        ----------
        X : np.ndarray
            2D telemetry feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Integer array of shape (n_samples,) containing cluster labels in the
            range [0, n_clusters - 1].
        """

        self._require_fit()
        X = self._validate_X(X)
        Xs = self._maybe_scale_transform(X)

        return self.model.predict(Xs)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for each sample.

        Higher scores (larger distances) imply greater anomaly likelihood.

        Parameters
        ----------
        X : np.ndarray
            2D array of telemetry feature vectors.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) containing distance-to-centroid scores.
            Higher distances typically correspond to more anomalous points.
        """

        self._require_fit()
        X = self._validate_X(X)
        Xs = self._maybe_scale_transform(X)
        distances = self.model.transform(Xs).min(axis=1)
        return distances


    def is_anomaly(self, X: np.ndarray, threshold: float | None = None):
        """
        Flag samples as anomalies based on their distance to nearest cluster.

        Parameters
        ----------
        X : np.ndarray
            2D array of telemetry feature vectors to evaluate.
        threshold : float, optional
        Distance cutoff for anomaly detection. Samples with scores greater than
        this value are considered anomalies. If None, the default threshold
        from the training data is used, based on `self.percentile`.

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_samples,) where True indicates an anomaly.
        """

        self._require_fit()
        X = self._validate_X(X)
        scores = self.score_samples(X)

        thr = threshold

        if thr is None:
            thr = self._default_threshold
            #If its somehow missing, in case any artifacts got stripped, get from current scores
            if thr is None:
                thr = np.percentile(scores, float(self.percentile))

        return scores > float(thr)