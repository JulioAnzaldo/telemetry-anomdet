# tests/test_models_smoke.py

"""
Smoke tests for supervised/unsupervised models.
Ensures basic fit/predict works and shapes line up.
"""

import numpy as np
from telemetry_anomdet.models.unsupervised import IsolationForestModel
from telemetry_anomdet.models.supervised import NaiveBayesClassifier

