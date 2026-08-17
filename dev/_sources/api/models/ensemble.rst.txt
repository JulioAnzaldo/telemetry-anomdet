Ensemble Model
==============

.. automodule:: telemetry_anomdet.models.ensemble
   :members:
   :undoc-members:
   :show-inheritance:

``AnomalyEnsemble`` wraps any set of ``BaseDetector`` subclasses and combines
their scores into a single ensemble score. All detectors receive 3D input
``(n_windows, window_size, n_features)``. Classical models flatten internally,
deep models consume it directly. The ensemble handles all routing.

``score_components()`` is the SHAP hook: it returns raw per-model scores before
normalization or combination, allowing ``SHAPExplainer`` to attribute the
ensemble decision back to individual sensor channels.

The ``combine`` strategy (``"mean"`` / ``"median"`` / ``"max"``) and
``normalize`` method (``"robust"`` / ``"minmax"`` / ``"none"``) are tuned on
the SMAP validation split in Phase 2. ``"robust"`` + ``"mean"`` is the default.
