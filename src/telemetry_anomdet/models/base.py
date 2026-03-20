# src/telemetry_anomdet/models/base.py

"""
BaseDetector: Shared interface for all anomaly detectors in  telemetry_anomdet.

Design
------
All detectors in this library follow PyOD convention.

    fit(x): Train the model on the provided data, sets post-fit attributes.
    predict(x): Return binary labels (0 for normal, 1 for anomaly).
    decision_function(x): Return anomaly scores (higher means more anomalous).
    is_anomaly(x): Boolean mask. Supports runtime thresholding override.

Input Convention
----------------
All detectors accept X of shape (n_windows, window_size, n_features). Classical detecrtors flatten X internally 
using feature_stat(). Sequence detectors (GDN, TranAD) consume X directly. 
The caller never maneges this distinction.

Post-fit Attributes
----------------
After fit(), every detectror exposes:

    decision_scores_: np.ndarray (n_windows,) Training anomaly scores.
    threshold_:       float                   Score cutoff from training.
    labels_:          np.ndarray (n_windows,) Binary labels from training (0 normal, 1 anomaly).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import pickle

import numpy as np

class BaseDetector(ABC):
    """
    Abstract base class for all anomaly detectors in telemetry-anomdet.

    Parameters
    ----------
    percentile : float, default = 95.0
        Percentile of training anomaly scores used to set ``threshold_``.
        For example, 95.0 means the top 5% most anomalous training windows
        are labelled as anomalies by default. Must be in (0, 100).

    Attributes (set after fit)
    --------------------------
    decision_scores_ : np.ndarray, shape (n_windows,)
        Anomaly scores on the training data. Higher values indicate greater
        anomaly likelihood. Set by ``_set_post_fit()`` at the end of ``fit()``.
    threshold_ : float
        Score cutoff derived from ``decision_scores_`` at ``self.percentile``.
        Used as the default decision boundary in ``predict()`` and
        ``is_anomaly()``.
    labels_ : np.ndarray, shape (n_windows,)
        Binary anomaly labels on the training data. 0 = normal, 1 = anomaly.
        Derived from ``decision_scores_`` and ``threshold_``.

    Notes
    -----
    Subclasses must implement ``fit()`` and ``decision_function()``.
    All other methods are provided and should not need to be overridden
    unless the detector has non-standard scoring behavior.
    """

    def __init__(self, percentile: float = 95.0):
        if not (0.0 < percentile < 100.0):
            raise ValueError(
                f"percentile must be in (0, 100), got {percentile!r}."
            )
        self.percentile = percentile

        # Post-fit attributes; None until fit() is called!
        self.decision_scores_: np.ndarray | None = None
        self.threshold_: float | None = None
        self.labels_: np.ndarray | None = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BaseDetector":
        """
        Train the detector on nominal telemetry windows.

        Subclasses must call ``self._set_post_fit(scores)`` at the end of
        this method, where ``scores`` are the training anomaly scores. This
        sets ``decision_scores_``, ``threshold_``, and ``labels_``
        automatically.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)
            Windowed telemetry tensor from ``windowify()``.
        y : np.ndarray, optional
            Ignored for unsupervised detectors. Present for API consistency.

        Returns
        -------
        self : BaseDetector
            Fitted detector instance, for method chaining.
        """

        ...
    
    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw anomaly scores for each input window.

        Higher scores indicate greater anomaly likelihood. This is the
        core scoring method. All other scoring methods call this one.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)
            Windowed telemetry tensor from ``windowify()``.

        Returns
        -------
        scores : np.ndarray, shape (n_windows,)
            Anomaly score per window.
        """

        ...
    
    # Core methods (implemented by subclass)
    def predict(self, X: np.ndarray):
        """
        Classify each window as normal (0) or anomalous (1).

        Uses ``threshold_`` from training by default. For runtime threshold
        adjustment, use ``is_anomaly()`` with a ``threshold`` or
        ``percentile`` override instead.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)
            Windowed telemetry tensor from ``windowify()``.

        Returns
        -------
        labels : np.ndarray, shape (n_windows,)
            Binary labels. 0 = normal, 1 = anomaly.
        """
        scores = self.decision_function(X)
        return (scores > self._get_threshold()).astype(int)
    
    def is_anomaly(self, X: np.ndarray, *, threshold: float | None = None, percentile: float | None = None,) -> np.ndarray:
        """
        Boolean anomaly mask with optional runtime threshold override.

        This is the human-in-the-loop hook. Operators can adjust sensitivity
        at inference time without retraining by passing a custom ``threshold``
        or ``percentile``. If neither is provided, the training derived
        ``threshold_`` is used.

        Priority: ``threshold`` > ``percentile`` > ``threshold_``.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)
            Windowed telemetry tensor from ``windowify()``.
        threshold : float, optional
            Fixed score cutoff. Windows with scores above this are anomalies.
        percentile : float, optional
            Compute the cutoff as this percentile of the current batch's
            scores. Useful when the operator wants to flag only the top N%
            of a given pass's windows rather than using training statistics.
            Must be in (0, 100).

        Returns
        -------
        mask : np.ndarray of bool, shape (n_windows,)
            True where the window is anomalous.

        Examples
        --------
        Default — use training threshold:
            flags = detector.is_anomaly(X3d)

        Tighten sensitivity (only top 2% flagged):
            flags = detector.is_anomaly(X3d, percentile=98.0)

        Fixed cutoff from operator feedback:
            flags = detector.is_anomaly(X3d, threshold=0.72)
        """
        scores = self.decision_function(X)

        if threshold is not None:
            thr = float(threshold)
        elif percentile is not None:
            if not (0.0 < percentile < 100.0):
                raise ValueError(
                    f"percentile must be in (0, 100), got {percentile!r}."
                )
            thr = float(np.percentile(scores, percentile))
        else:
            thr = self._get_threshold()

        return scores > thr
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Alias for ``decision_function()``.

        Provided for compatibility with code that uses the sklearn/PyOD
        ``score_samples`` convention. Prefer ``decision_function()`` for
        new code.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, window_size, n_features)

        Returns
        -------
        scores : np.ndarray, shape (n_windows,)
        """
        return self.decision_function(X)
    
    def _set_post_fit(self, scores: np.ndarray) -> None:
        """
        Set the three standard post-fit attributes from training scores.

        Call this at the end of ``fit()`` in every subclass:

            scores = self._compute_my_scores(X)
            self._set_post_fit(scores)
            return self

        Parameters
        ----------
        scores : np.ndarray, shape (n_windows,)
            Anomaly scores on the training data.
        """
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1:
            raise ValueError(
                f"_set_post_fit() expects a 1D score array, got shape {scores.shape}."
            )
        self.decision_scores_ = scores
        self.threshold_ = float(np.percentile(scores, self.percentile))
        self.labels_ = (scores > self.threshold_).astype(int)
    
    def _get_threshold(self) -> float:
        """
        Return the training derived threshold, raising if not yet fitted.
        """
        if self.threshold_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )
        return self.threshold_
    
    def _require_fit(self) -> None:
        """
        Raise if the detector has not been fitted.

        Use at the top of ``decision_function()`` in subclasses:

            def decision_function(self, X):
                self._require_fit()
                X = self._validate_X(X)
                ...
        """
        if self.threshold_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )
    
    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and coerce input to a clean 3D float array.

        Enforces the (n_windows, window_size, n_features) input convention.
        Call at the top of both ``fit()`` and ``decision_function()``.

        Parameters
        ----------
        X : array like
            Input to validate.

        Returns
        -------
        X : np.ndarray, shape (n_windows, window_size, n_features)
            Validated array, coerced to float64.

        Raises
        ------
        ValueError
            If X is not 3D, has 0 windows, or contains non-finite values.
        """
        X = np.asarray(X, dtype = float)

        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D input (n_windows, window_size, n_features), "
                f"got shape {X.shape}. "
                f"Pass the direct output of windowify()."
            )
        if X.shape[0] == 0:
            raise ValueError(
                "Input X has 0 windows. Check that windowify() received "
                "enough data for at least one window."
            )
        if X.shape[1] == 0:
            raise ValueError("Input X has window_size = 0.")
        if X.shape[2] == 0:
            raise ValueError("Input X has 0 features (channels).")
        if not np.isfinite(X).all():
            n_bad = (~np.isfinite(X)).sum()
            raise ValueError(
                f"Input X contains {n_bad} non-finite value(s) (NaN or Inf). "
                f"Run interpolate_gaps() and normalize_fit() before windowing."
            )
        return X
    
    def save(self, path: str | Path) -> None:
        """
        Serialize the fitted detector to disk using pickle.

        Parameters
        ----------
        path : str or Path
            Destination file path. Conventionally ends in ``.pkl``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str | Path) -> "BaseDetector":
        """
        Deserialize a detector from disk.

        Parameters
        ----------
        path : str or Path
            Path to a pickled detector file created by ``save()``.

        Returns
        -------
        detector : BaseDetector
            The deserialized, fitted detector instance.

        Notes
        -----
        Only load files from trusted sources.
        """
        path = Path(path)
        with path.open("rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        fitted = self.threshold_ is not None
        params = self._get_params()
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return (
            f"{self.__class__.__name__}({param_str}, "
            f"fitted={fitted})"
        )
    
    def _get_params(self) -> dict:
        """
        Return constructor parameters for ``__repr__``.

        Subclasses can override this to include their own hyperparameters.
        By default returns ``{"percentile": self.percentile}``.
        """
        return {"percentile": self.percentile}