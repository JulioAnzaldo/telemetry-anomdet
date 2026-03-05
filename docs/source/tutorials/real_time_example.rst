Real-Time Anomaly Detection Example
====================================

This tutorial walks through the intended end-to-end workflow for a ground station
pass: fitting the ensemble on nominal telemetry, then running inference on a new
pass and flagging anomalous windows.

All detectors accept 3D input of shape ``(n_windows, window_size, n_features)`` —
the direct output of ``windowify()``. You never reshape data manually per detector.

Training on Nominal Telemetry
------------------------------

Train the ensemble once on a nominal baseline pass. No anomaly labels are used —
all detectors learn a model of normal behavior and score deviations from it at
inference time.

.. code-block:: python

    import numpy as np
    from telemetry_anomdet.models.unsupervised.pca import PCAAnomaly
    from telemetry_anomdet.models.unsupervised.kmeans import KMeansAnomaly
    from telemetry_anomdet.models.ensemble import AnomalyEnsemble

    # X_train: (n_windows, window_size, n_features) — nominal pass, no anomaly labels
    # Produced by: make_feature_table() → windowify()
    X_train = np.random.default_rng(0).normal(size=(500, 50, 25))

    ensemble = AnomalyEnsemble(
        models={
            "pca":    PCAAnomaly(n_components=10),
            "kmeans": KMeansAnomaly(n_clusters=5, scale=True),
        },
        combine="mean",
        normalize="robust",
        percentile=95.0,
    )

    ensemble.fit(X_train)
    # Post-fit: ensemble.threshold_, ensemble.decision_scores_, ensemble.labels_ are set

Inference on a New Pass
------------------------

Pass new telemetry windows through the fitted ensemble. Use ``is_anomaly()`` for
flagging, or ``decision_function()`` to get the raw scores for downstream processing.

.. code-block:: python

    # X_test: new pass, same shape contract
    X_test = np.random.default_rng(1).normal(size=(100, 50, 25))

    # Continuous anomaly scores — higher means more anomalous
    scores = ensemble.decision_function(X_test)

    # Boolean flags using the training-derived threshold
    flags = ensemble.is_anomaly(X_test)

    # Operator override: tighten sensitivity at runtime without retraining
    flags_strict = ensemble.is_anomaly(X_test, percentile=98.0)
    flags_custom  = ensemble.is_anomaly(X_test, threshold=0.75)

    print(f"Flagged {flags.sum()} / {len(flags)} windows as anomalous")

Per-Model Score Decomposition (SHAP hook)
------------------------------------------

``score_components()`` returns raw per-model scores before normalization or
combination. This is the input to ``SHAPExplainer`` (Phase 3): SHAP perturbs
input channels and measures how each sensor drives each model's score independently,
producing per-channel attribution weights for every anomalous window.

.. code-block:: python

    components = ensemble.score_components(X_test)
    # {"pca": array(100,), "kmeans": array(100,)}

    for name, model_scores in components.items():
        print(f"{name}: mean={model_scores.mean():.4f}, max={model_scores.max():.4f}")

Coming in Phase 3+
-------------------

Once ``SHAPExplainer`` and the LLM reasoning layer are implemented, the workflow
extends to:

.. code-block:: python

    # Phase 3 — per-channel attribution (not yet implemented)
    # shap_values = explainer.explain(ensemble, X_anomalous)
    # {"sensor_01": 0.42, "sensor_02": 0.18, ...}

    # Phase 4 — LLM diagnostic report (not yet implemented)
    # report = llm_engine.explain(shap_values, context, channel_names)
    # DiagnosticReport(
    #     anomaly_type="power subsystem fault",
    #     severity="high",
    #     primary_channels=["sensor_01", "sensor_07"],
    #     explanation="...",
    #     recommended_action="...",
    #     confidence=0.87,
    # )