# src/telemetry_anomdet/models/supervised/bayes.py

"""
Simple supervised models (examples).

This file contains a small Naive Bayes classifier wrapper that follows the package
BaseModel style (fit/predict).
"""

from __future__ import annotations
from sklearn.naive_bayes import GaussianNB
from dataclasses import dataclass
from ..base import BaseModel

@dataclass
class NaiveBayesClassifier(BaseModel):
    """
    Gaussian Naive Bayes classifier wrapper.

    Parameters:
    config:
        Optional dictionary passed to the underlying sklearn classifier.
    """