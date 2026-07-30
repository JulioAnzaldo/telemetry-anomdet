# src/telemetry_anomdet/models/__init__.py

"""
Models package for telemetry_anomdet.

Subpackages:
- supervised: models that require labels (classification/regression).
- unsupervised: anomaly detection algorithms that operate without labels.
- deep: sequence-aware detectors (GDN, ...) requiring the optional deep extra.
"""

from . import deep, supervised, unsupervised

__all__ = ["deep", "supervised", "unsupervised"]
