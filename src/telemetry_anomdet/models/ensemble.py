# src/telemetry_anomdet/models/ensemble.py

"""
Ensemble combinator for anomaly detectors.

This module defines an `AnomalyEnsemble` that can wrap multiple
BaseModel compatible detectors and combine their scores into a 
single ensemble score and anomaly decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Iterable

import numpy as np

from .base import BaseModel


@dataclass
class AnomalyEnsemble(BaseModel):
    """
    Ensemble of anomaly detection models.

    Parameters
    ----------
    models : Mapping[str, BaseModel]
        Dictionary mapping model names to fitted or unfitted model instances.
        Example:
            {
                "kmeans": KMeansAnomaly(n_clusters = 5),
                "pca": PCAAnomaly(n_components = 5),
            }
    combine : str, default = "mean"
        Strategy to combine normalized scores across models.
        Options:
            - "mean"   : arithmetic mean of scores
            - "median" : median score
            - "max"    : maximum score
    normalize : str, default = "robust"
        How to normalize per-model scores before combining:
            - "none"   : use raw scores
            - "minmax" : (x - min) / (max - min)
            - "robust" : (x - median) / IQR, clipped to [0, 1]
        Normalization statistics are estimated from the training data.
    percentile : float, default=95.0
        Percentile of ensemble training scores used as default anomaly
        threshold in `is_anomaly()`. For example, 95.0 means the top 5%
        most anomalous training windows are considered anomalies.

    Notes
    -----
    - All wrapped models are expected to follow the `BaseModel` API.
    - For scoring, the ensemble will prefer a model's `score_samples(X)`
      method if it exists; otherwise it falls back to `predict(X)`.
    """

    models: Mapping[str, BaseModel]
    combine: str = "mean"
    normalize: str = "robust"
    percentile: float = 95.0

    # Learned normalization + threshold artifacts
    _norm_stats: Dict[str, dict] = field(default_factory = dict, init = False)
    _default_threshold: Optional[float] = field(default = None, init = False)

    # Helpers
    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Input X must be 2D (n_samples, n_features), got {X.shape}")
        if X.shape[0] == 0:
            raise ValueError("Input X has 0 rows")
        if not np.isfinite(X).all():
            raise ValueError("Input X contains NaN or infinite values")
        return X

    def _score_one_model(self, name: str, model: BaseModel, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores from a single model.

        Prefers `score_samples(X)` if available, else falls back to `predict(X)`.
        """
        if hasattr(model, "score_samples"):
            scores = model.score_samples(X)
        else:
            scores = model.predict(X)

        scores = np.asarray(scores)
        if scores.ndim != 1:
            raise ValueError(
                f"Model '{name}' returned scores with shape {scores.shape}, "
                "expected 1D array of shape (n_samples,)."
            )
        return scores

    def _compute_norm_stats(self, scores: np.ndarray, method: str) -> dict:
        """
        Compute normalization statistics for a single model's scores.
        """
        if method == "none":
            return {}

        if method == "minmax":
            s_min = float(np.min(scores))
            s_max = float(np.max(scores))
            # Protect against degenerate case
            if s_max == s_min:
                s_max = s_min + 1e-12
            return {"min": s_min, "max": s_max}

        if method == "robust":
            med = float(np.median(scores))
            q1 = float(np.percentile(scores, 25.0))
            q3 = float(np.percentile(scores, 75.0))
            iqr = q3 - q1
            if iqr <= 0.0:
                iqr = 1e-12
            return {"median": med, "iqr": iqr}

        raise ValueError(f"Unknown normalization method: {method!r}")

    def _apply_norm(self, scores: np.ndarray, stats: dict, method: str) -> np.ndarray:
        """
        Apply normalization to a score vector using stored statistics.
        Returns a float array of the same shape.
        """
        scores = np.asarray(scores, dtype=float)

        if method == "none":
            return scores

        if method == "minmax":
            s_min = stats["min"]
            s_max = stats["max"]
            out = (scores - s_min) / (s_max - s_min)
            return np.clip(out, 0.0, 1.0)

        if method == "robust":
            med = stats["median"]
            iqr = stats["iqr"]
            # Scale so that approx [Q1, Q3] ~ [0, 1]
            out = 0.5 + (scores - med) / (2.0 * iqr)
            return np.clip(out, 0.0, 1.0)

        raise ValueError(f"Unknown normalization method: {method!r}")

    def _combine_matrix(self, S: np.ndarray) -> np.ndarray:
        """
        Combine per-model score matrix into a single ensemble score.

        Parameters
        ----------
        S : np.ndarray
            2D array of shape (n_models, n_samples), already normalized
            or in the desired scale.

        Returns
        -------
        np.ndarray
            1D array of shape (n_samples,) with combined scores.
        """
        if S.ndim != 2:
            raise ValueError(f"Score matrix must be 2D (n_models, n_samples), got {S.shape}")

        if self.combine == "mean":
            return np.mean(S, axis=0)
        if self.combine == "median":
            return np.median(S, axis=0)
        if self.combine == "max":
            return np.max(S, axis=0)

        raise ValueError(f"Unknown combine strategy: {self.combine!r}")

    # Core API
    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """
        Fit all underlying models and estimate normalization / thresholds.

        Parameters
        ----------
        X : np.ndarray
            2D telemetry feature matrix (n_samples, n_features).
        y : np.ndarray, optional
            Optional labels (for supervised models). Most anomaly models will
            ignore this.
        """
        X = self._validate_X(X)

        # Fit each model (if they aren't already fitted).
        # We don't try to be too clever here: just call fit.
        for name, model in self.models.items():
            if y is None:
                model.fit(X)
            else:
                model.fit(X, y)

        # Compute training scores per model
        per_model_scores = {}
        for name, model in self.models.items():
            scores = self._score_one_model(name, model, X)
            per_model_scores[name] = scores

        # Learn normalization stats
        self._norm_stats = {}
        for name, scores in per_model_scores.items():
            self._norm_stats[name] = self._compute_norm_stats(scores, self.normalize)

        # Build normalized score matrix and combined ensemble scores on training data
        S = []
        for name, scores in per_model_scores.items():
            stats = self._norm_stats.get(name, {})
            S.append(self._apply_norm(scores, stats, self.normalize))
        S = np.vstack(S)  # shape: (n_models, n_samples)

        ensemble_scores = self._combine_matrix(S)

        # Default threshold from training distribution
        p = float(self.percentile)
        if not (0.0 < p < 100.0):
            raise ValueError("percentile must be in (0, 100)")
        self._default_threshold = float(np.percentile(ensemble_scores, p))

        return self

    def score_components(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get per model scores for the given samples (before combination).

        Returns
        -------
        dict
            Mapping model_name -> 1D np.ndarray of shape (n_samples,).
        """
        X = self._validate_X(X)

        scores = {}
        for name, model in self.models.items():
            scores[name] = self._score_one_model(name, model, X)
        return scores

    def score_samples(self, X: np.ndarray, *, normalize: Optional[bool] = True) -> np.ndarray:
        """
        Compute ensemble anomaly scores for each sample.

        Parameters
        ----------
        X : np.ndarray
            2D telemetry feature matrix.
        normalize : bool, optional
            If True (default), apply the configured normalization method
            (`self.normalize`) before combining. If False, use raw scores
            as returned by each model.

        Returns
        -------
        np.ndarray
            1D array of ensemble scores. Higher values correspond to
            greater anomaly likelihood.
        """
        X = self._validate_X(X)
        per_model = self.score_components(X)

        S = []
        for name, scores in per_model.items():
            if normalize and self.normalize != "none":
                stats = self._norm_stats.get(name)
                if stats is None:
                    raise RuntimeError(
                        f"No normalization stats found for model '{name}'. "
                        "Did you call `fit()` on the ensemble?"
                    )
                S.append(self._apply_norm(scores, stats, self.normalize))
            else:
                S.append(np.asarray(scores, dtype=float))

        S = np.vstack(S)  # (n_models, n_samples)
        return self._combine_matrix(S)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ensemble anomaly scores.

        This mirrors the unsupervised convention in the rest of the package,
        where `predict()` returns a score, not a class label.
        """
        return self.score_samples(X)

    def is_anomaly(self, X: np.ndarray, *, threshold: float | None = None, percentile: float | None = None) -> np.ndarray:
        """
        Flag samples as anomalies based on ensemble scores.

        Parameters
        ----------
        X : np.ndarray
            2D telemetry feature matrix.
        threshold : float, optional
            Fixed score cutoff. Samples with scores greater than this
            value are considered anomalies.
        percentile : float, optional
            If provided, override the default `self.percentile` and
            compute a threshold based on the given percentile of the
            current batch's ensemble scores.

        Returns
        -------
        np.ndarray
            Boolean array of shape (n_samples,) where True indicates an anomaly.
        """
        scores = self.score_samples(X)

        thr: float
        if threshold is not None:
            thr = float(threshold)
        else:
            if percentile is not None:
                p = float(percentile)
                if not (0.0 < p < 100.0):
                    raise ValueError("percentile must be in (0, 100)")
                thr = float(np.percentile(scores, p))
            else:
                if self._default_threshold is None:
                    # Fall back to using `self.percentile` on current scores
                    p = float(self.percentile)
                    thr = float(np.percentile(scores, p))
                else:
                    thr = self._default_threshold

        return scores > thr