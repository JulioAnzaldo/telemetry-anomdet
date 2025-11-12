import numpy as np
from telemetry_anomdet.models.unsupervised.kmeans import KMeansAnomaly

def test_kmeans_anomaly_basic():
    rng = np.random.default_rng(0)
    X0 = rng.normal([0,0], 0.3, size=(200,2))
    X1 = rng.normal([3,3], 0.3, size=(200,2))
    X_train = np.vstack([X0, X1])

    X_out = rng.normal([8,8], 0.3, size=(10,2))
    X_test = np.vstack([X_train[:20], X_out])

    det = KMeansAnomaly(n_clusters=2, scale=True, percentile=95.0).fit(X_train)

    labels = det.predict(X_test)
    assert set(labels).issubset({0, 1})

    scores = det.score_samples(X_test)
    assert np.all(scores >= 0)

    flags = det.is_anomaly(X_test)
    print("Flagged last 10 outliers:", flags[-10:].sum(), "/ 10")  # <-- add this
    assert flags[-10:].sum() >= 7

