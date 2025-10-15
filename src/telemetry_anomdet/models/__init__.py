# src/telemetry_anomdet/models/__init__.py

"""
Models package for telemetry_anomdet.

Subpackages:
- supervised: models that require labels (classification/regression).
- unsupervised: anomaly detection algorithms that operate without labels.
"""

from . import supervised, unsupervised

__all__ = ["supervised", "unsupervised"]
