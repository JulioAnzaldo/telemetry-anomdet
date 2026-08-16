# src/telemetry_anomdet/ingest/__init__.py

"""
Ingest subpackage: dataset loaders and parsers.
"""

from .csv_loader import load_from_csv
from .dataset import TelemetryDataset
from .smap import (
    anomaly_point_mask,
    load_smap,
    load_smap_channel,
    load_smap_labels,
)

# ccsds_loader is deliberately absent: it sits behind the optional ccsds extra,
# and importing it here would make the base install fail without ccsdspy.
__all__ = [
    "TelemetryDataset",
    "load_from_csv",
    "load_smap",
    "load_smap_channel",
    "load_smap_labels",
    "anomaly_point_mask",
]
