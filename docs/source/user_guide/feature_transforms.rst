Choosing a Feature Transform
============================

``KANGDN`` builds its forecast from two things: each channel's own recent
history, and its neighbours in the learned graph. ``feat_mode`` selects how the
first of those enters the network, and on a single channel it is the only path
that does anything at all, because a graph with one node is a self-loop.

The figures here are a snapshot of three benchmarks rather than a property of
the toolkit. They are kept in the documentation, not in module docstrings, so
they can be revised as the detectors change.


The modes
---------

``linear``
    ``nn.Linear(window, embed_dim)``. The default, and what both ``GDN`` and
    ``KANGDN`` have always used. Raw telemetry enters through a single linear
    map, so the KAN layers only ever see an already-compressed embedding.

``kan``
    ``KAN(x)``. A nonlinearity over each channel's own window, with none of the
    window discarded.

``ar_kan``
    ``KAN(a * x)``. AR-KAN as published by Wu et al. (2025), arXiv:2509.02967:
    an autoregressive model is fitted in closed form by Yule-Walker, its
    coefficients are frozen, and they scale the window element-wise before a KAN
    sees it. The AR stage adds no trainable parameters and no backward pass.

``ar_kan_residual``
    ``Linear(x) + KAN(a * x)``. The AR-KAN branch corrects a linear baseline
    instead of replacing it, so the unfiltered window always reaches the encoder.


Which one wins depends on the input, and it reverses
----------------------------------------------------

Event-level score at each model's best label-free operating point. SMAP columns
are event F1 on 55 channels; the ESA column is event-wise F0.5 on Mission1
channels 41 to 46, which is that benchmark's own headline metric.

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - ``feat_mode``
     - SMAP, 1 channel
     - SMAP, 25 channels
     - ESA-ADB, 6 channels
   * - ``linear``
     - 0.683
     - 0.541
     - 0.674
   * - ``kan``
     - 0.720
     - 0.504
     - 0.657
   * - ``ar_kan``
     - 0.583
     - 0.446
     - **0.745**
   * - ``ar_kan_residual``
     - **0.724**
     - **0.597**
     - 0.705

The published AR-KAN formulation is worst on SMAP and best on ESA-ADB. That is
not noise, and the mechanism is measurable.

Spacecraft telemetry is strongly persistent, so Yule-Walker concentrates almost
all of its weight on the most recent lag. Measured on real SMAP channels, the
lag-1 coefficient carries 93 to 100 percent of the filter's energy, which means
nearly every one of the ``window`` inputs to the KAN is multiplied by
approximately zero. The transform sees the previous sample rather than the
window.

Whether that is fatal depends entirely on whether anything else can supply the
missing history:

* A SMAP record is one real sensor plus 24 near-constant command flags. The
  graph has nothing informative to relate, so a node stripped of its own past
  cannot recover it from anywhere, and ``ar_kan`` comes last.
* ESA-ADB's channels 41 to 46 are six sensors from one subsystem with a median
  cross-channel correlation of 0.86. Neighbours supply exactly the context the
  filter discards, and the aggressive focus on the newest sample becomes an
  advantage.

The same reversal shows in the class breakdown on SMAP: ``ar_kan`` detects 6 of
26 contextual anomalies against 10 for ``linear``, and contextual anomalies are
precisely the ones that require history rather than magnitude.


Recommendation
--------------

**Prefer** ``ar_kan_residual`` **when the input's structure is unknown.** Its
linear branch carries the whole window regardless of what the filter does, so
the AR-KAN branch can only add. It wins both SMAP configurations and comes third
on ESA-ADB, and it is the only mode that is never worse than second.

Use ``ar_kan`` when the channels are known to be genuinely related, which is the
case it is best in and the case onboard deployment targets.

Use ``linear`` when the model has to reach hardware today: it is the only mode
:func:`~telemetry_anomdet.models.deep.codegen.generate_c` can emit. Distillation
to NumPy supports every mode.


Cost
----

Any KAN mode multiplies the feature transform's coefficient count by roughly
``grid_size + spline_order``, and at small ``embed_dim`` that transform dominates
the distilled flash footprint. Measured on a single-channel model at
``embed_dim=8``, ``window=50``, ``grid_size=5``:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - ``feat_mode``
     - Parameters
     - Flash, float32
   * - ``linear``
     - 1,096
     - 4.3 KB
   * - ``kan``
     - 4,288
     - 16.8 KB
   * - ``ar_kan``
     - 4,288 + 50 frozen
     - 16.9 KB
   * - ``ar_kan_residual``
     - 4,696 + 50 frozen
     - 18.5 KB

The AR filter itself is nearly free: ``n_nodes * window`` frozen floats, 200
bytes for a single channel, with no gradient and no training cost. The expense
is the KAN that follows it.

Window length matters as much as width here, because the transform's input is
the window. The same ``ar_kan_residual`` model falls from 18.5 KB to 4.5 KB at
``embed_dim=4``, ``window=30``, ``grid_size=3``, which was its best measured
configuration on SMAP rather than a deliberate shrink.


A mode that was removed
-----------------------

``Linear(x) + KAN(x)``, briefly offered as ``kan_residual``, is gone. It was
measured in six regimes across three datasets and was never best in any of them.
On ESA-ADB it merely tied the plain ``linear`` transform it exists to improve on,
while identifying affected channels worse.

It is recorded here because the obvious question about the table above is
whether the AR filter contributes anything at all, or whether the residual
structure is doing all the work. ``kan_residual`` is the control that answers it:
at matched settings ``ar_kan_residual`` beat it in all four SMAP and MSL
comparisons. The filter earns its place.
