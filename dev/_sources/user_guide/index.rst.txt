User Guide
==========

How the toolkit fits together, and the reasoning behind the choices that are
not obvious from the API alone.

Start with the pipeline overview for the shape of the whole thing. Read
:doc:`anomaly_scoring` before quoting any benchmark number: it explains why the
usual SMAP metric rates uniform random noise above every trained detector, and
which figures to trust instead. :doc:`feature_transforms` covers choosing
between the deep detectors, where the best option reverses depending on whether
the input's channels are genuinely related.

.. toctree::
   :maxdepth: 2

   pipeline_overview
   anomaly_scoring
   feature_transforms
   real_time_integration
   glossary
