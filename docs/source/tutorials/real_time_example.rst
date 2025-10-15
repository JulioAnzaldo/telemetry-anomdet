Real-Time Anomaly Detection Example
===================================

This tutorial shows a minimal example of processing streaming telemetry data in real time.

Simulated Streaming Data
------------------------

.. code-block:: python

    from telemetry_anomdet.preprocessing import preprocessing
    from telemetry_anomdet.feature_extraction import features
    from telemetry_anomdet.models.unsupervised import IsolationForestModel

    # Simulate a small telemetry data stream
    stream = [
        {"sensor1": 0.5, "sensor2": 1.2},
        {"sensor1": 0.6, "sensor2": 1.1},
        {"sensor1": 5.0, "sensor2": -2.0},  # simulated anomaly
    ]

    # Initialize preprocessing and model
    model = IsolationForestModel(config={"n_estimators": 50, "contamination": 0.1})

    # Convert streaming data into a small batch
    import numpy as np

    X = np.array([[d["sensor1"], d["sensor2"]] for d in stream])

    # Fit model (in real use, you'd train it on baseline / normal data first)
    model.fit(X)

    # Predict anomaly scores
    scores = model.predict(X)

    for data_point, score in zip(stream, scores):
        print(f"Input: {data_point}, Anomaly Score: {score:.4f}")