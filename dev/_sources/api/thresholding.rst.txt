Thresholding Module
===================

Turning a continuous error signal into alarms, without labels.

A detector emits a score per timestep; operations need a boolean. The selectors
here choose that cutoff from the score distribution alone, so no anomaly labels
are required at deployment time. ``threshold_for_budget`` caps the fraction of
points flagged, which is the control operations can actually state;
``dynamic_threshold`` needs no budget chosen in advance.

For which selector to use and how the defaults here differ from the published
telemanom method, see :doc:`../user_guide/anomaly_scoring`.

.. automodule:: telemetry_anomdet.thresholding
   :members:
