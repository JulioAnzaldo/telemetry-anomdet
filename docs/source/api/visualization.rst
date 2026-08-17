Visualization
=============

Figures for inspecting a fitted detector. These are deliberately not general
time-series plots, which matplotlib already does better. Each one shows
something only this toolkit can show: what the graph detector learned, and which
channels drove a score.

matplotlib is optional and lives in the ``viz`` extra:

.. code-block:: bash

   uv add "telemetry-anomdet[viz]"

Every function draws onto an ``Axes`` and returns it, so figures compose into
larger layouts. Nothing is shown or written to disk on your behalf.

Reading the learned graph
-------------------------

:func:`~telemetry_anomdet.visualization.plot_sensor_graph` is the most direct
check on whether the graph found real structure, which the metrics show only
indirectly.

On data whose records hold a single true sensor beside command flags, every node
must still choose ``topk`` neighbours, so the graph fills with arbitrary edges
and contributes nothing. Where channels are genuinely related, the coupled pairs
appear as the thick dark reciprocal edges, because edge shading follows the
cosine similarity of the learned embeddings.

Passing one window's deviations turns the figure from a picture of the model
into an explanation of a single alarm:

.. code-block:: python

   scores = detector.decision_function(X)
   worst = scores.argmax()

   ax = plot_sensor_graph(
       detector,
       channel_names=names,
       deviations=detector.channel_deviations(X)[worst],
   )

.. automodule:: telemetry_anomdet.visualization
   :members:
   :undoc-members:
   :show-inheritance:
