Unsupervised Models
===================

All unsupervised detectors inherit from :class:`~telemetry_anomdet.models.base.BaseDetector`
and accept 3D input ``(n_windows, window_size, n_features)`` — the direct output of
``windowify()``. Classical models flatten internally via ``features_stat()``. The caller
never manages this conversion.

PCAAnomaly
----------

.. automodule:: telemetry_anomdet.models.unsupervised.pca
   :members:
   :undoc-members:
   :show-inheritance:

KMeansAnomaly
-------------

.. automodule:: telemetry_anomdet.models.unsupervised.kmeans
   :members:
   :undoc-members:
   :show-inheritance:

IsolationForestModel
--------------------

.. note::
   ``IsolationForestModel`` is under active development and will be completed in Phase 1
   (SMAP Integration). It will wrap ``sklearn.ensemble.IsolationForest`` as a full
   ``BaseDetector`` subclass and be registered in ``AnomalyEnsemble`` alongside
   ``PCAAnomaly`` and ``KMeansAnomaly``.

.. automodule:: telemetry_anomdet.models.unsupervised.isolation_forest
   :members:
   :undoc-members:
   :show-inheritance: