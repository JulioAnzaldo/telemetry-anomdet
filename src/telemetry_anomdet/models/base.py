# src/telemetry_anomdet/models/base.py

"""
Shared BaseModel for all models in telemetry_anomdet.

Shared functionality goes here (save/load, config handling).
"""

from __future__ import annotations
import numpy as np

class BaseModel():
    """
    Abstract base model for consistent API across supervised & unsupervised models.
    """