BaseDetector
============

``BaseDetector`` is the abstract base class all detectors in telemetry-anomdet inherit from.
It enforces a consistent interface modeled on PyOD convention.

**Abstract methods** (subclasses must implement):

- ``fit(X, y=None)`` — train on nominal telemetry windows; must call ``_set_post_fit(scores)`` before returning
- ``decision_function(X)`` — return raw anomaly scores, shape ``(n_windows,)``; higher = more anomalous

**Provided methods** (inherited free by all subclasses):

- ``predict(X)`` — binary labels (0 = normal, 1 = anomaly) using ``threshold_``
- ``is_anomaly(X, *, threshold=None, percentile=None)`` — boolean mask with optional runtime threshold override
- ``score_samples(X)`` — alias for ``decision_function()``
- ``save(path)`` / ``load(path)`` — pickle serialization

**Post-fit guarantee**: after ``fit()``, every detector always exposes:

- ``decision_scores_`` — training anomaly scores, shape ``(n_windows,)``
- ``threshold_`` — score cutoff at ``self.percentile`` of training scores
- ``labels_`` — binary training labels derived from ``decision_scores_`` and ``threshold_``

.. automodule:: telemetry_anomdet.models.base
   :members:
   :undoc-members:
   :show-inheritance: