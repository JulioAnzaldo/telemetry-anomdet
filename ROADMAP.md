Telemetry Anomaly Detection - Roadmap (MVP)

1. Ingestion
- Implemented csv_loader
  - Load telemetry CSVs
  - Normalize to long form (timestamp, variable, value)
  - Detect and convert wide to long if needed

2. Preprocessing (Long Form)
- clean(): remove non existent values, non numeric values, and physically impossible values
- dedupe(): drop retransmitted or duplicate rows
- integrity_check(): verify UTC timestamps and sorting
- resample(): align all sensors to a uniform cadence (~5s)
- interpolate_gaps(): fill small gaps (like missing datapoints from sensors, only for tiny gaps)
- normalize_fit() / normalize_apply(): optional scaling step

Output: Clean, aligned, long form telemetry

3. Feature Extraction (Wide Form)
- pivot_wide(): reshape long to wide form (columns = sensors)
- windowify(): slice into fixed windows (~50-100 samples)
- features_stat(): compute simple statistics per variable (mean, standard deviation, min, max)
- make_feature_table(): build final feature matrix X

Output: Feature table ready for models

4. Modeling (Unsupervised Bayes)
- Implement Gaussian Naive Bayes density model
  - Fit on all (assumed normal) data
  - Compute log likelihood for each window (how likely data is to be normal or anomaly)
  - Low probability = anomaly
- Generate CSV:
  window_id, t_start, t_end, anomaly_score, is_anomaly (boolean)

Output: Ranked anomaly scores per window

5. Evaluation and Visualization
- Plot anomaly scores vs. time
- Highlight top anomalies on voltage/temperature plots
- Confirm low-score windows align with real behavior

6. Future Work (post class probably, or if we have time)
- Add Isolation Forest and One-Class SVM models
- Create models/base.py interface for multi-model scoring
- Implement ensemble scoring (mean, max, weighted. Using scores from multiple models)
- Automate result exports and threshold tuning

Deliverable (MVP)
- End-to-end pipeline runs on UHF dataset from OPS-SAT (OPS-SAT is a 3U CubeSat launched by the European Space Agency (ESA))
- Produces cleaned features and anomaly scores
- Functional unsupervised Bayes model

General foot notes
- Our project applies AI concepts of reasoning under uncertainty and probabilistic learning to detect anomalies in satellite telemetry data (or other spacecraft). 
- We use a Gaussian Naive Bayes model, which learns the normal distribution of each sensor feature (voltage, current, temperature) and computes the likelyhood of new observations under those learned distributions.
- The model assumes that features are conditionally independent even though telemetry variables are often correlated. 
- Each feature is modeled as a Gaussian (normal distribution), and the joint probability of a telemetry window is the product of these individual probabilities. 
- Windows with very low likelyhoods are flagged as anomalies, representing states unlikely under normal operation.