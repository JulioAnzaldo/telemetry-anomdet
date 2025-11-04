Telemetry Anomaly Detection Toolkit
===================================

Welcome to the **telemetry-anomdet** documentation.

This toolkit provides a modular pipeline for ingesting, preprocessing, feature extraction,
and unsupervised anomaly detection in spacecraft or ground based telemetry.
It supports CSV, CCSDS, or HDF5 input formats and can be integrated into real-time systems
through Kafka or batch processing pipelines.

Current features include:
- CSV loader with long/wide format detection
- Data cleaning, deduplication, and resampling
- Statistical feature extraction over time windows
- Gaussian Naive Bayes based anomaly scoring
- Extensible model interface for multi-model ensembles

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   user_guide/pipeline_overview
   user_guide/real_time_integration
   user_guide/glossary
   tutorials/real_time_example
   applications/cubesat_ops

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/ingest
   api/preprocessing
   api/feature_extraction
   api/models/base
   api/models/ensemble
   api/models/supervised
   api/models/unsupervised
   api/models/visualization