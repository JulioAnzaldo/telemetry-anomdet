# src/telemetry_anomdet/models/ensemble.py

"""
Ensemble combinator for anomaly detectors.

This module defines an `AnomalyEnsemble` that can wrap multiple
BaseDetector compatible detectors and combine their scores into a 
single ensemble score and anomaly decision.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from .base import BaseDetector

class AnomalyEnsemble(BaseDetector):
    """
    Stacking ensemble of anomaly detectors.

    Parameters
    ----------
    models : Mapping[str, BaseDetector]
        Named detectors to include in the ensemble. Example:

            {
                "pca":    PCAAnomaly(n_components = 10),
                "iso":    IsolationForestAnomaly(),
                "gdn":    GDN(epochs = 50),
                "tranad": TranAD(window_size = 50),
            }

    combine : str, default = "mean"
        Strategy for combining normalized per-model scores.
        One of ``"mean"``, ``"median"``, ``"max"``.
    normalize : str, default = "robust"
        Per-model score normalization before combining.
        One of ``"none"``, ``"minmax"``, ``"robust"``.
        Statistics are estimated from training data.
    percentile : float, default = 95.0
        Percentile of ensemble training scores used as the default
        anomaly threshold. Passed to ``BaseDetector._set_post_fit()``.

    Attributes (set after fit)
    --------------------------
    decision_scores_ : np.ndarray, shape (n_windows,)
        Ensemble anomaly scores on training data.
    threshold_ : float
        Default anomaly cutoff derived from training scores at ``percentile``.
    labels_ : np.ndarray, shape (n_windows,)
        Binary anomaly labels on training data. 0 = normal, 1 = anomaly.
    """

    def __init__(self, models: Mapping[str, BaseDetector], combine: str = "mean", normalize: str = "robust", percentile: float = 95.0,):
        super().__init__(percentile = percentile)
        self.models = models
        self.combine = combine
        self.normalize = normalize

        self._norm_stats: Dict[str, dict] = {}

    # ---- Helpers ----
    def _compute_norm_stats(self, scores: np.ndarray, method: str) -> dict:
        """
        Compute normalization statistics from training scores.

        Called once during ``fit()``. The returned dict is stored in
        ``_norm_stats`` and reused at inference time by ``_apply_norm()``,
        ensuring test scores are scaled on the same training distribution.

        ``"robust"`` is preferred over ``"minmax"`` for anomaly detection
        because anomalies are extreme values by definition: they would skew
        a minmax scale and compress the normal operating range.

        Parameters
        ----------
        scores : np.ndarray, shape (n_windows,)
            Raw training scores from a single detector.
        method : str
            One of ``"none"``, ``"minmax"``, ``"robust"``.

        Returns
        -------
        dict
            Statistics needed to apply the same normalization to new data.
        """

        if method == "none":
            return {}
        if method == "minmax":
            s_min = float(np.min(scores))
            s_max = float(np.max(scores))
            if s_max == s_min:
                s_max = s_min + 1e-12
            return {"min": s_min, "max": s_max}
        if method == "robust":
            med = float(np.median(scores))
            q1  = float(np.percentile(scores, 25.0))
            q3  = float(np.percentile(scores, 75.0))
            iqr = q3 - q1
            if iqr <= 0.0:
                iqr = 1e-12
            return {"median": med, "iqr": iqr}
        raise ValueError(f"Unknown normalization method: {method!r}")

    def _apply_norm(self, scores: np.ndarray, stats: dict, method: str) -> np.ndarray:
        """
        Apply normalization to a score array using training statistics.

        For ``"robust"``, the formula ``0.5 + (x - median) / (2 * IQR)``
        maps the middle 50% of training scores to approximately ``[0, 1]``.
        Scores outside that range are clipped: genuine anomalies saturate
        at 1.0, which is the desired behavior.

        Parameters
        ----------
        scores : np.ndarray
            Raw scores from a single detector (training or inference).
        stats : dict
            Statistics returned by ``_compute_norm_stats()``.
        method : str
            Must match the method used to compute ``stats``.

        Returns
        -------
        np.ndarray
            Normalized scores clipped to ``[0, 1]``.
        """

        scores = np.asarray(scores, dtype = float)
        if method == "none":
            return scores
        if method == "minmax":
            out = (scores - stats["min"]) / (stats["max"] - stats["min"])
            return np.clip(out, 0.0, 1.0)
        if method == "robust":
            out = 0.5 + (scores - stats["median"]) / (2.0 * stats["iqr"])
            return np.clip(out, 0.0, 1.0)
        raise ValueError(f"Unknown normalization method: {method!r}")

    def _combine_matrix(self, S: np.ndarray) -> np.ndarray:
        """
        Combine per-model score matrix into ensemble scores.

        Parameters
        ----------
        S : np.ndarray, shape (n_models, n_windows)

        Returns
        -------
        np.ndarray, shape (n_windows,)
        """
        if self.combine == "mean":
            return np.mean(S, axis = 0)
        if self.combine == "median":
            return np.median(S, axis = 0)
        if self.combine == "max":
            return np.max(S, axis = 0)
        raise ValueError(f"Unknown combine strategy: {self.combine!r}")

    # ---- BaseDetector interface ----
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AnomalyEnsemble":
        """
        Fit all detectors and estimate normalization statistics.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)
            Windowed telemetry tensor from windowify().
        y : ignored

        Returns
        -------
        self : AnomalyEnsemble
        """
        X = self._validate_X(X)

        for _, model in self.models.items():
            model.fit(X) if y is None else model.fit(X, y)

        # Per-model training scores
        per_model = {name: model.decision_function(X) for name, model in self.models.items()}

        # Learn normalization stats from training scores
        self._norm_stats = {
            name: self._compute_norm_stats(scores, self.normalize)
            for name, scores in per_model.items()
        }

        # Build normalized score matrix to ensemble scores
        S = np.vstack([
            self._apply_norm(scores, self._norm_stats[name], self.normalize)
            for name, scores in per_model.items()
        ])
        ensemble_scores = self._combine_matrix(S)

        self._set_post_fit(ensemble_scores)
        return self

    def decision_function(self, X: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        """
        Compute ensemble anomaly scores.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)
        normalize : bool, default=True
            Apply configured normalization before combining.

        Returns
        -------
        scores : np.ndarray, shape (n_windows,)
            Higher = more anomalous.
        """
        self._require_fit()
        X = self._validate_X(X)

        per_model = self.score_components(X)

        S = []
        for name, scores in per_model.items():
            if normalize and self.normalize != "none":
                stats = self._norm_stats.get(name)
                if stats is None:
                    raise RuntimeError(
                        f"No normalization stats for model '{name}'. "
                        "Call fit() on the ensemble first."
                    )
                S.append(self._apply_norm(scores, stats, self.normalize))
            else:
                S.append(np.asarray(scores, dtype = float))

        return self._combine_matrix(np.vstack(S))

    # ---- SHAP hook ----
    def score_components(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Per-model raw anomaly scores before combination.

        This is the input to SHAPExplainer; SHAP perturbs X and measures
        how each channel affects each model's score independently.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)

        Returns
        -------
        dict
            ``{model_name: np.ndarray of shape (n_windows,)}``
        """
        X = self._validate_X(X)
        return {name: model.decision_function(X) for name, model in self.models.items()}

    # ---- repr ----
    def _get_params(self) -> dict:
        return {
            "models":    list(self.models.keys()),
            "combine":   self.combine,
            "normalize": self.normalize,
            "percentile": self.percentile,
        }