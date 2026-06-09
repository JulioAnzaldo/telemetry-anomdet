# src/telemetry_anomdet/ingest/ccsds_loader.pyv

from __future__ import annotations

import pandas as pd

from telemetry_anomdet.ingest.dataset import TelemetryDataset


def load_from_ccsds(file_or_stream) -> TelemetryDataset:
    """
    Load telemetry from a CCSDS file or stream.
    """

    df = pd.DataFrame()  # Placeholder
    # TODO: Implement CCSDS packet parsing logic

    return TelemetryDataset(df)
