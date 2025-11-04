Glossary
========

.. glossary::

   Long form
      Each row represents a single data point: (timestamp, variable, value).  
      Used for ingest and preprocessing because it generalizes well across sensors.

   Wide form
      Each row represents one timestamp, and each column is a variable.  
      Used for feature extraction and model input.

   Window
      A fixed-size block of sequential telemetry samples (50–100).  
      Each window becomes one feature vector for the model.

   Anomaly score
      A numeric measure of how unlikely a telemetry window is under the learned model.  
      Low probability means high anomaly score.

   Feature
      A numerical summary (mean, standard deviation, min, max, etc.) computed from each window.

   Gaussian Naive Bayes
      A probabilistic model assuming each feature follows a normal distribution and  
      that all features are conditionally independent given the system’s state.

   Conditional independence
      The assumption that once the system’s overall state is known,  
      the behavior of one sensor does not affect another.

   CCSDS
      Consultative Committee for Space Data Systems; a standardized packet protocol  
      used by many satellite missions for telemetry and telecommand.

   Kafka
      A distributed event-streaming platform used for real-time data pipelines.  
      In this project, it enables streaming telemetry and anomaly scores.