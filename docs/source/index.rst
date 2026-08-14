Telemetry Anomaly Detection Toolkit
====================================

**telemetry-anomdet** is an open-source anomaly detection toolkit for spacecraft telemetry.
It ingests raw telemetry (SMAP, CSV), preprocesses it, and runs classical and graph-based
deep detectors behind a single interface, selecting alarm thresholds without labels. A
trained detector distills to Power of Ten conformant C, so the same model that is
evaluated on the ground can run on flight hardware.

Per-channel SHAP attribution and LLM-generated diagnostic reports, aimed at producing
actionable diagnostics within the ground station inter-pass window, are on the roadmap.

Benchmarked on SMAP (NASA). MSL (NASA) and ESA-ADB (ESA) results land in v0.3.0:
SMAP and MSL are univariate per record, while ESA-ADB is the genuinely multivariate
benchmark that exercises the inter-sensor graph.

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
- MSL results, using the existing SMAP loader (``spacecraft = "MSL"``) (v0.3.0)
- ESA-ADB evaluation on genuinely multivariate telemetry (v0.3.0)


Contents
--------

- :doc:`getting_started` - install and first run
- :doc:`user_guide/index` - how the pipeline fits together, and which metric to trust
- :doc:`tutorials/real_time_example` - end-to-end worked example
- :doc:`applications/index` - CubeSat operations and onboard deployment
- :doc:`api/index` - every public function and class

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   user_guide/index
   tutorials/index
   applications/index
   api/index