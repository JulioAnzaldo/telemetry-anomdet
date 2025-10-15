# src/telemetry_anomdet/models/unsupervised/isolation_forest.py
"""
Isolation Forest unsupervised anomaly detector wrapper.

This module exposes a simple class with a consistent API:
- fit(X)
- predict(X) -> anomaly scores (higher = more anomalous)
- save / load via BaseModel
"""

from __future__ import annotations
from sklearn.ensemble import IsolationForest
from dataclasses import dataclass
from ..base import BaseModel

@dataclass
class IsolationForestModel(BaseModel):
    """
    Wrapper for sklearn IsolationForest.

    Parameters
    ----------
    config:
        Optional dict to pass to sklearn IsolationForest.
    """