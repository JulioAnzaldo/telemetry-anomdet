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

        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2025-01-01T00:00:00Z", "2025-01-01T00:01:00Z"], utc = True
                ),
                "variable": ["temp", "pressure"],
                "value": [25.5, 1013.25],
            }
        ).sort_values(["timestamp", "variable"], ignore_index = True)

        return cls(df)
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Return a copy of the Dataframe to prevent accidental mutation.
        """

        return self._df.copy()
    
    @property
    def data(self) -> pd.DataFrame:
        """
        Direct access to the underlying DataFrame.
        """
        return self._df
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Head of the underlying DataFrame (copy).
        """
        
        return self._df.head(n).copy()
    
    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        cols = list(self._df.columns)
        return f"TelemetryDataset(n={len(self)}, cols={cols})"