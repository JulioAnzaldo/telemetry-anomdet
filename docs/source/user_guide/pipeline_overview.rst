Pipeline Overview
=================

This page summarizes the full telemetry anomaly detection workflow.

1. **Ingestion** — Load telemetry from CSV or CCSDS into a long-form ``TelemetryDataset``
2. **Preprocessing** — Clean, deduplicate, resample, interpolate gaps, normalize (``pipeline()``)
3. **Feature Extraction** — Pivot to wide form, slide windows via ``windowify()``, producing a 3D tensor ``(n_windows, window_size, n_features)``
4. **Detection** — Pass the 3D tensor to ``AnomalyEnsemble.fit()`` (training) or ``decision_function()`` / ``is_anomaly()`` (inference). The ensemble combines scores from ``PCAAnomaly``, ``KMeansAnomaly``, and ``IsolationForestModel`` (classical) and, in Phase 2, ``GDN`` and ``TranAD`` (deep). All detectors share the ``BaseDetector`` interface; the caller never adapts per model.
5. **Attribution** — ``score_components()`` returns per-model raw scores. ``SHAPExplainer`` (Phase 3) perturbs input channels to produce per-channel attribution weights.
6. **Diagnosis** — The LLM reasoning layer (Phase 4) receives SHAP attributions and generates a structured ``DiagnosticReport`` with anomaly type, severity, primary channels, explanation, and recommended action.