Deep Models
===========

Deep detectors inherit from :class:`~telemetry_anomdet.models.base.BaseDetector`
and share the same ``fit`` / ``decision_function`` / ``predict`` / ``is_anomaly``
API as the classical models, so they can be stacked or swapped without glue.

These models require ``torch``, which is an optional dependency:

.. code-block:: bash

   uv add "telemetry-anomdet[deep]"

GDN
---

Graph deviation network. Learns a sensor embedding, builds a top-k relational
graph over channels, and forecasts each channel from its graph neighbours. The
anomaly score is the deviation between forecast and observation, normalised per
node by the training median and IQR.

``score_channels`` restricts which channels may contribute to that deviation.
Channels excluded from the score still feed the model as context. This matters
on data that mixes continuous sensors with discrete mode and status flags, where
a command switch is a state change rather than a fault.

.. autoclass:: telemetry_anomdet.models.deep.gdn.GDN
   :members:
   :undoc-members:
   :show-inheritance:

KANGDN
------

GDN with the multilayer perceptrons replaced by Kolmogorov-Arnold layers, whose
learned univariate splines are what make the distillation and C generation below
possible.

.. autoclass:: telemetry_anomdet.models.deep.kan_gdn.KANGDN
   :members:
   :undoc-members:
   :show-inheritance:

Distillation
------------

Extracts a fitted KANGDN into plain NumPy: spline coefficients, the frozen
adjacency, and the per-node normalisation constants. The result evaluates
without torch, which is the intermediate step toward the flight artifact.

.. automodule:: telemetry_anomdet.models.deep.distill
   :members:
   :undoc-members:
   :show-inheritance:

C Code Generation
-----------------

Emits a distilled detector as Power of Ten conformant C with no dynamic
allocation, alongside golden vectors for host-side conformance testing. See
:doc:`../../applications/onboard_deployment` for the target hardware, the
resource budget, and how responsibility is split between ground and flight.

.. automodule:: telemetry_anomdet.models.deep.codegen
   :members:
   :undoc-members:
   :show-inheritance:
