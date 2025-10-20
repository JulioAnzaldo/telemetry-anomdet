# src/telemetry_anomdet/ingest/dataset.py

"""
Telemetry dataset loader.

This module a lightweight class called TelemetryDataset that wraps telemetry data in a consistent form factor for downstream processing.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

@dataclass
class TelemetryDataset:
    """
    Container for telemetry data.
    """

    # Pandas DataFrame containing telemetry rows.
    _df: pd.DataFrame
    
    @classmethod
    def synthetic(cls) -> Self:
        """
        Generate a small synthetic telemetry dataset for development / testing.

        Parameters:
        N/A for now.

        Returns:
        TelemetryDataset
        """

        # TODO: Implement function (will do once everything else is implemented)
        data = {'timestamp': pd.to_datetime(['2025-01-01', '2025-01-01']),
                'variable': ['temp', 'pressure'],
                'value': [25.5, 1013.25]}
        df = pd.DataFrame(data)

        return cls(df)
    
    @classmethod
    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a copy of the Dataframe to prevent accidental mutation.
        """

        return self._df.copy()