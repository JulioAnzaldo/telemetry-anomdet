Telemetry Anomaly Detection Toolkit
====================================

**telemetry-anomdet** is an open-source anomaly detection toolkit for spacecraft telemetry.
It ingests raw telemetry (CCSDS, CSV), preprocesses it, and runs a stacking ensemble of
classical and deep learning detectors with per-channel SHAP attribution and LLM-generated
diagnostic reports — designed to produce actionable diagnostics within the ground station
inter-pass window.

Validated on SMAP (NASA) and OPS-SAT (ESA).

Current features:

- CCSDS and CSV ingestion into long-form ``TelemetryDataset``
- Preprocessing pipeline: clean, dedupe, resample, interpolate gaps, normalize
- Windowed feature extraction: statistical features and raw 3D tensors for sequence models
- ``BaseDetector`` interface: unified ``fit`` / ``decision_function`` / ``predict`` / ``is_anomaly`` API shared by all detectors
- ``PCAAnomaly`` and ``KMeansAnomaly`` classical detectors (3D input, flatten internally)
- ``AnomalyEnsemble``: stacking combinator with configurable normalization and combine strategy
- Per-model score decomposition via ``score_components()`` (SHAP hook)

Coming in the next few months:

- ``IsolationForestAnomaly``
- ``GDN`` — graph deviation network for inter-sensor relational anomalies
- ``TranAD`` — transformer-based sequence reconstruction
- ``SHAPExplainer`` — per-channel attribution over ``score_components()``
- SMAP ingestion and benchmark pipeline

Coming in by the end of 2026:

- LLM reasoning layer (Llama 3.1 8B on Jetson Orin via llama.cpp)
- OPS-SAT cross-dataset generalization evaluation


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   user_guide/pipeline_overview
   user_guide/real_time_integration
   user_guide/glossary
   tutorials/real_time_example
   applications/cubesat_ops

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/ingest
   api/preprocessing
   api/feature_extraction
   api/models/base
   api/models/ensemble
   api/models/unsupervised