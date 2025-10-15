# src/telemetry_anomdet/ingest/csv_loader.py

from __future__ import annotations
from telemetry_anomdet.ingest.dataset import TelemetryDataset
import pandas as pd

def load_from_csv(path: str) -> TelemetryDataset:
    """
        Load telemetry from a CSV file.

        Notes:
        CSV must contain at least colums: timestamp, variable, value. timestamp will be converted to pandas datetime.

        Parameters:
        path - Path to CSV file.

        Returns:
        TelemetryDataset
        """
    
    df = pd.read_csv(path)
    # TODO: Add logic to convert timestamp and validate columns

    return TelemetryDataset(df)