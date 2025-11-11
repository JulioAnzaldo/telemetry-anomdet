# src/telemetry_anomdet/models/base.py

"""
Shared BaseModel for all models in telemetry_anomdet.

Defines a unified interface for model configuration, fitting,
prediction, and optional persistence.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pickle

class BaseModel():
    """
    Abstract base model for consistent API across supervised & unsupervised models.

    Notes
    -----
    - `config` is a free form dictionary for hyperparameters or settings.
    - Subclasses are responsible for implementing `fit` and `predict`.
    - Optional methods like `score_samples` or `is_anomaly` can be added
      by subclasses for anomaly detection use cases.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize the model with an optional configuration dictionary.

        Parameters
        ----------
        config : dict, optional
            Hyperparameters or other settings for this model instance.
        """
        self.config = config or {}

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        """
        Train the model (to be implemented by subclass).

        Parameters
        ----------
        X : np.ndarray
            Input data. For most models this is a 2D array of shape
            (n_samples, n_features).
        y : np.ndarray, optional
            Target labels for supervised models. For unsupervised models
            this is usually None.
        """

        raise NotImplementedError
    
    # Core methods (implemented by subclass)
    def predict(self, X: np.ndarray):
        """
        Predict outputs for the given input data.

        For supervised models, this typically returns class labels.
        For clustering models, this might return cluster indices.

        Parameters
        ----------
        X : np.ndarray
            Input data to predict on.
        """

        raise NotImplementedError
    
    def save(self, path: str | Path):
        """
        Save the trained model to disk using pickle.

        Parameters
        ----------
        path : str or Path
            Destination file path.

        Notes
        -----
        - Uses Python pickle; loading untrusted files is unsafe.
        - Subclasses should ensure all attributes are picklable.
        """

        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str | Path):
        """
        Load a previously saved model instance from disk.

        Parameters
        ----------
        path : str or Path
            Path to the pickled model file.

        Returns
        -------
        BaseModel
            The deserialized model instance.

        Notes
        -----
        - Assumes the file was created via `BaseModel.save()` or an equivalent.
        - Only load files from trusted sources.
        """

        path = Path(path)
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        cname = self.__class__.__name__
        return f"<{cname} config={self.config}>"