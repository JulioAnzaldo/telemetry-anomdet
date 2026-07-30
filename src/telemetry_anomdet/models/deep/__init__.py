# src/telemetry_anomdet/models/deep/__init__.py

"""
Deep, sequence-aware anomaly detection models.

These detectors consume the 3D windowed tensor directly (no feature flattening)
and require the optional ``deep`` extra (PyTorch)::

    uv sync --extra deep

Importing this subpackage does not import torch; torch is imported lazily when a
detector is fitted, so the base install can still import the models package.
"""

from .gdn import GDN

__all__ = ["GDN"]
