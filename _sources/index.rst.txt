Telemetry Anomaly Detection Toolkit
====================================

**telemetry-anomdet** is an open-source anomaly detection toolkit for spacecraft telemetry.
It ingests raw telemetry (SMAP, MSL, ESA-ADB, CSV), preprocesses it, and runs classical
and graph-based deep detectors behind a single interface, selecting alarm thresholds
without labels and pointing each alarm at the channels that raised it. A trained detector
distills to Power of Ten conformant C, so the same model that is evaluated on the ground
can run on flight hardware.

SHAP attribution and LLM-generated diagnostic reports, aimed at producing actionable
diagnostics within the ground station inter-pass window, are on the roadmap.

Benchmarked on SMAP and MSL (NASA) and ESA-ADB (ESA). SMAP and MSL are univariate per
record, while ESA-ADB is the genuinely multivariate benchmark that exercises the
inter-sensor graph.

Current features:

Ingestion and preprocessing

- SMAP, MSL and CSV ingestion into long-form ``TelemetryDataset`` (``load_smap``, ``load_smap_channel``, ``load_smap_labels``, ``load_from_csv``)
- ESA-ADB ingestion straight from the official archive, with the benchmark's own splits (``load_esa``, ``load_esa_labels``, ``esa_splits``). Its payloads are pickled, so load only that archive, never untrusted files
- Preprocessing pipeline: clean, dedupe, resample, interpolate gaps, normalize
- Windowed feature extraction: statistical features and raw 3D tensors for sequence models

Detection

- ``BaseDetector`` interface: unified ``fit`` / ``decision_function`` / ``predict`` / ``is_anomaly`` API shared by all detectors
- ``PCAAnomaly`` and ``KMeansAnomaly`` classical detectors (3D input, flatten internally)
- ``GDN``: graph deviation network forecasting each channel from its learned top-k neighbours
- ``KANGDN``: GDN with Kolmogorov-Arnold layers, the form the flight artifact is distilled from, with ``feat_mode`` selecting the node feature transform
- ``score_channels``: restrict which channels may raise an alarm while all of them still feed the model
- Per-channel attribution (``channel_deviations``, ``dominant_channels``) and the learned sensor graph (``learned_graph``, ``plot_sensor_graph``)
- ``AnomalyEnsemble``: stacking combinator with configurable normalization and combine strategy
- Per-model score decomposition via ``score_components()`` (SHAP hook)

Scoring, thresholding, evaluation

- Label-free operating point selection (``threshold_for_budget``, ``dynamic_threshold``) with sequence post-processing
- Point-adjusted and event-level evaluation (``point_adjusted_f1``, ``evaluate_sequences``, ``pr_auc``, ``false_alarm_rate_at_recall``)
- A reproducible SMAP and MSL benchmark reporting both, with a random baseline row (``examples/smap_benchmark.py``)
- An ESA-ADB benchmark following that dataset's protocol, with F0.5 as the headline metric (``examples/esa_benchmark.py``)

Onboard deployment

- Distillation of a fitted ``KANGDN`` to a torch-free NumPy evaluator
- Power of Ten conformant C generation with golden vectors for every ``feat_mode``, plus host and ESP32-S3 targets

Coming next:

- ``IsolationForestAnomaly``
- ``TranAD``: transformer-based sequence reconstruction
- ``SHAPExplainer``: per-channel attribution over ``score_components()``

Coming in by the end of 2026:

- LLM reasoning layer (Llama 3.1 8B on Jetson Orin via llama.cpp)


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