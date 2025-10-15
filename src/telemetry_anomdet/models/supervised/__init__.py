# src/telemetry_anomdet/models/supervised/__init__.py

"""
Supervised models (labelled data).
"""

from .bayes import NaiveBayesClassifier

__all__ = ["NaiveBayesClassifier"]
