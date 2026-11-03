API Reference
=============

Every public function and class, grouped by the stage of the pipeline it
belongs to.

Detectors share one interface, described in :doc:`models/base`: ``fit`` /
``decision_function`` / ``predict`` / ``is_anomaly``, with ``decision_scores_``,
``threshold_`` and ``labels_`` set after fitting. Classical and deep detectors
are therefore interchangeable.

Data
----

.. toctree::
   :maxdepth: 2

   ingest
   preprocessing
   feature_extraction

Detection
---------

.. toctree::
   :maxdepth: 2

   models/base
   models/unsupervised
   models/deep
   models/ensemble

Scoring
-------

.. toctree::
   :maxdepth: 2

   thresholding
   evaluation
