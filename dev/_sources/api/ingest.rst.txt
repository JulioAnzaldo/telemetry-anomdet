Ingestion
=========

This module contains functions for loading telemetry data from various sources.
Every loader returns the same long-form ``TelemetryDataset``, so downstream
preprocessing and detection do not care where the data came from.

.. automodule:: telemetry_anomdet.ingest.smap
   :members:

.. automodule:: telemetry_anomdet.ingest.csv_loader
   :members:

.. automodule:: telemetry_anomdet.ingest.ccsds_loader
   :members:

.. autoclass:: telemetry_anomdet.ingest.dataset.TelemetryDataset
   :members: