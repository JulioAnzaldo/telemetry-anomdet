# telemetry_anomdet/models.py

"""
Model definitions and training logic.

This module provides classes for fitting and predicting
anomalies in telemetry data.
"""

class DummyModel:
    """Minimal model stub for import and test purposes."""

    def fit(self, X, y = None):
        """Pretend to fit model; return self."""
        return self

    def predict(self, X):
        """Return zero predictions (placeholder)."""
        return [0] * len(X)
