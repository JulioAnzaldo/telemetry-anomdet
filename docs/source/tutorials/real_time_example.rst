Real-Time Anomaly Detection Example
===================================

This tutorial shows a minimal example of processing streaming telemetry data in real time.

Simulated Streaming Data
------------------------

.. code-block:: python

    from telemetry_anomdet import preprocessing, features, models

    # Simulate a small data stream
    stream = [
        {"sensor1": 0.5, "sensor2": 1.2},
        {"sensor1": 0.6, "sensor2": 1.1},
    ]

    # Initialize a dummy model
    model = models.DummyModel()

    for data_point in stream:
        cleaned = preprocessing.clean([data_point])
        feats = features.extract_features(cleaned)
        prediction = model.predict(feats)
        print(f"Input: {data_point}, Prediction: {prediction}")