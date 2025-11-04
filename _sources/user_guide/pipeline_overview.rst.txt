Pipeline Overview
=================

This page summarizes the full telemetry anomaly detection workflow.

1. Ingestion – Load and normalize telemetry (CSV, CCSDS, or HDF5)
2. Preprocessing – Clean, deduplicate, resample, fill gaps
3. Feature Extraction – Convert to wide form, window, compute statistics
4. Modeling – Apply Gaussian Naive Bayes to detect anomalies
5. Visualization – Plot scores and highlight anomalies