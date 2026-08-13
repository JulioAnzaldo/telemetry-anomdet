Telemetry Anomaly Detection Toolkit
====================================

**telemetry-anomdet** is an open-source anomaly detection toolkit for spacecraft telemetry.
It ingests raw telemetry (SMAP, CSV), preprocesses it, and runs a stacking ensemble of
classical and deep learning detectors with per-channel SHAP attribution and LLM-generated
diagnostic reports, designed to produce actionable diagnostics within the ground station
inter-pass window.

Validated on SMAP (NASA), with OPS-SAT (ESA) as a cross-dataset generalization source.

Current features:

Ingestion and preprocessing

- SMAP and CSV ingestion into long-form ``TelemetryDataset`` (``load_smap``, ``load_smap_channel``, ``load_smap_labels``, ``load_from_csv``)
- Preprocessing pipeline: clean, dedupe, resample, interpolate gaps, normalize
- Windowed feature extraction: statistical features and raw 3D tensors for sequence models

Detection

- ``BaseDetector`` interface: unified ``fit`` / ``decision_function`` / ``predict`` / ``is_anomaly`` API shared by all detectors
- ``PCAAnomaly`` and ``KMeansAnomaly`` classical detectors (3D input, flatten internally)
- ``GDN``: graph deviation network forecasting each channel from its learned top-k neighbours
- ``KANGDN``: GDN with Kolmogorov-Arnold layers, the form the flight artifact is distilled from
- ``score_channels``: restrict which channels may raise an alarm while all of them still feed the model
- ``AnomalyEnsemble``: stacking combinator with configurable normalization and combine strategy
- Per-model score decomposition via ``score_components()`` (SHAP hook)

Scoring, thresholding, evaluation

- Label-free operating point selection (``threshold_for_budget``, ``dynamic_threshold``) with sequence post-processing
- Point-adjusted and event-level evaluation (``point_adjusted_f1``, ``evaluate_sequences``, ``pr_auc``, ``false_alarm_rate_at_recall``)
- A reproducible SMAP benchmark reporting both, with a random baseline row (``examples/smap_benchmark.py``)

Onboard deployment

- Distillation of a fitted ``KANGDN`` to a torch-free NumPy evaluator
- Power of Ten conformant C generation with golden vectors, plus host and ESP32-S3 targets

Coming next:

- ``IsolationForestAnomaly``
- ``TranAD``: transformer-based sequence reconstruction
- ``SHAPExplainer``: per-channel attribution over ``score_components()``

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
   user_guide/anomaly_scoring
   user_guide/glossary
   tutorials/real_time_example
   applications/cubesat_ops
   applications/onboard_deployment

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/ingest
   api/preprocessing
   api/feature_extraction
   api/evaluation
   api/thresholding
   api/models/base
   api/models/ensemble
   api/models/unsupervised
   api/models/deep