# src/telemetry_anomdet/models/supervised/bayes.py

"""
Gaussian Naive Bayes

Each feature is assumed to follow a Gaussian (normal) distribution conditioned on the class label y.
"""

from __future__ import annotations
from dataclasses import dataclass
from ..base import BaseModel
import numpy as np

@dataclass
class GaussianNaiveBayes(BaseModel):
    """
    Gaussian Naive Bayes classifier.

    Attributes
    ----------
    classes : np.ndarray
        Unique class labels observed during training.
    mean : dict
        Per-class feature means.
    var : dict
        Per-class feature variances.
    priors : dict
        Class prior probabilities.
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Estimate mean, variance, and prior for each class.
        """
        
        pass

    def _pdf(self, class_idx, x: np.ndarray):
        """
        Compute the Gaussian probability density for a given class.
        """

        pass

    def _posterior(self, x: np.ndarray):
        """
        Compute the posterior probability for each class.
        """

        pass

    def predict(self, X: np.ndarray):
        """
        Predict class label for each sample.
        """

        pass