Scoring and Thresholding
========================

How detector output is turned into alarms, which metric that should be judged
by, and what the two choices measured on the SMAP benchmark.

The figures here are a snapshot of one benchmark rather than a property of the
toolkit. They are kept in the documentation, not in module docstrings, so they
can be revised as the detectors change.


Why the metric matters
----------------------

The same detector configuration can score 0.71 or 0.13 depending only on how
its output is counted, so the metric has to be settled before any result means
anything.

Point-adjusted F1
    The convention most SMAP results are reported under. Every point of a
    labelled segment counts as detected if any single point inside it was
    flagged. It was introduced so that a detector is not punished for flagging
    an event slightly late, which is reasonable, but it overcorrects: a long
    true detection can outweigh hundreds of short false ones.

Event level
    A labelled anomaly counts once if any prediction overlaps it, and each
    prediction overlapping nothing counts once against precision. This is what
    an operator experiences, because what reaches them is a count of alarms.
    See :func:`telemetry_anomdet.evaluation.evaluate_sequences`.

The gap between them is not academic. Scored on SMAP, a detector emitting
uniform random scores reaches roughly 0.75 to 0.99 point-adjusted F1, above
every trained configuration measured, while scoring 0.000 at the event level.
SMAP's anomaly segments are long, so flagging a small fraction of points at
random clips nearly all of them.

The benchmark therefore reports both, leads with the event-level columns, and
includes a ``random@all`` row so the floor is always visible.


What moved the numbers
----------------------

Measured on 54 SMAP channels at the event level, with a threshold chosen
without labels. Ordered by how much each change was worth.

.. list-table::
   :header-rows: 1
   :widths: 46 18 36

   * - Change
     - Event F1
     - Note
   * - Starting point: score the maximum deviation over all 25 channels
     - 0.126
     - 245 false alarms
   * - Score the telemetry channel only
     - 0.427
     - The other 24 are command flags
   * - Forecast the telemetry channel alone, no graph
     - 0.620
     - Specific to SMAP, see below
   * - Widen the threshold candidate floor from 2.5 to 1.0 sigma
     - 0.724
     - Validated on held-out channels

For reference, telemanom reports precision 0.838, recall 0.899 and F1 0.867 on
the same 54 channels, from 62 true positives, 12 false positives and 7 false
negatives over 69 labelled anomalies. The configuration above reaches precision
0.793, recall 0.667 and F1 0.724, with the same 12 false positives. The
remaining difference is recall.


Choosing what may raise an alarm
--------------------------------

The largest single effect came from restricting which channels contribute to
the deviation score, through ``score_channels`` on the detector.

A SMAP record is one telemetry dimension plus 24 one-hot command flags, and the
labels describe the telemetry. Taking the maximum deviation over all 25 raises
an alarm whenever a command switches, which is a state change rather than a
fault.

The general form of this applies to real telemetry too, which mixes continuous
sensors with discrete status and mode channels. Those channels should feed the
model, because they carry context that improves the forecast, while not being
allowed to raise alarms on their own.


A note on the univariate result
-------------------------------

Dropping the graph and forecasting the telemetry channel alone improves
precision, recall and false alarms simultaneously on SMAP. That is a statement
about this dataset, not about the architecture.

A SMAP record contains one real sensor, so there are no inter-sensor
relationships for a graph to represent. GDN was designed for and validated on
plant data with 51 and 127 genuine sensors, and the anomalies that matter most
in flight are exactly the ones a single channel cannot show: readings that are
each individually plausible but wrong in relation to each other. The
multivariate graph remains the target for onboard deployment.

The benchmark's ``kangdn@telemetry`` row exists as a control that isolates the
graph's contribution on this dataset, not as a recommended configuration.


Selecting an operating point
----------------------------

Two label-free selectors are provided in
:mod:`telemetry_anomdet.thresholding`.

``threshold_for_budget``
    Caps the fraction of points flagged, as an upper bound rather than a
    target: fewer are flagged when the signal cannot resolve the fraction
    asked for. That suits a deployed trigger, because operations can state an
    alarm rate but cannot state a recall they have no way to observe.

``dynamic_threshold``
    Chooses a candidate maximising the reduction it produces in the mean and
    standard deviation of the error signal, against the number of points and
    sequences it flags. Needs no budget chosen in advance.

Two defaults depart from the published method.

The candidate floor is 1.0 sigma rather than 2.5. Deviation scores are already
normalised by each node's training median and IQR, so their distribution is
much tighter than the smoothed prediction errors the original method was tuned
on. Tuning the floor on half the channels and measuring on the other half, over
40 held-out folds, gave event-level F1 0.696 against 0.590, better in every
fold. Anything between 0.5 and 2.0 lands on the same plateau, and above 2.5 the
score falls away sharply.

Pruning is off. It cost recall without buying precision in every configuration
measured.

Sliding the threshold over windows, flooring it with a global level, merging
nearby runs and filtering short ones were all tested. Each helped while the
score was taken over all 25 channels, and none helped once it was not, which is
the same lesson as above: what feeds the threshold matters more than how the
threshold is chosen.

One interaction is worth knowing. Below roughly a 3 percent budget the flagged
points are scattered singletons that never form a run of two, so
``filter_sequences`` with its default minimum length discards all of them.


Channels with no error spread
-----------------------------

A deviation score divides each channel's forecast error by that channel's own
training median and IQR, which is what makes the channels comparable. A channel
that never moved during training breaks it: its error IQR is exactly zero, so
the division returns a number set by the guard epsilon rather than by the data.
On SMAP three of 25 channels are in this state, all of them command flags that
sit constant for long stretches, and their deviations reached 1.7e10. Since a
window's score is the maximum across channels, those three decided every score
in the run, which is the mechanism behind the 0.126 event-level F1 measured when
every channel was allowed to raise an alarm.

``GDN`` therefore replaces the spread of such channels with a fraction of their
own median error, bringing the maximum deviation to 1.29e4, within 1.5 times the
most extreme healthy channel, and leaving 19 of 25 channels driving scores
instead of a handful. ``degenerate_channels_`` reports which channels were
affected, ``fit`` warns and names them when they can actually raise an alarm,
and both graph figures mark them.

Detecting these channels and repairing them are separate decisions, and
conflating them is a mistake worth recording. An earlier version used a single
ratio of 0.02 for both, on the reasoning that no real channel shows spread that
small relative to its own median error. That holds across the 25-channel input
and fails on the single-channel one, where the telemetry channel has an IQR of
3.6e-4 against a median error of 1.8e-2: a ratio of exactly 0.02, so the repair
reached a channel whose spread was real and changed its scores. Detection now
triggers only at a spread of zero to within floating point noise, while the
replacement value stays at 0.02, which it is free to do because it reaches
nothing else.
