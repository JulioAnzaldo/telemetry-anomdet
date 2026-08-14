<h2 align="center">telemetry-anomdet</h2>

<p align="center">
  <a href="https://julioanzaldo.github.io/telemetry-anomdet/"><img alt="Documentation" src="https://img.shields.io/badge/Docs-Online-blue"></a>
  <a href="https://test.pypi.org/project/telemetry-anomdet/"><img alt="Test PyPI" src="https://img.shields.io/badge/Test-PyPI-yellow"></a>
  <a href="https://pypi.org/project/telemetry-anomdet/"><img alt="PyPI" src="https://img.shields.io/pypi/v/telemetry-anomdet?label=PyPi"></a>
  <a href="https://github.com/JulioAnzaldo/telemetry-anomdet/actions/workflows/ci.yml"><img alt="Testing" src="https://img.shields.io/github/actions/workflow/status/JulioAnzaldo/telemetry-anomdet/ci.yml?branch=main&label=Testing&logo=github&logoColor=white"></a>
  <a href="https://github.com/JulioAnzaldo/telemetry-anomdet/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue"></a>
</p>

**telemetry-anomdet** is an open-source anomaly detection toolkit for spacecraft telemetry. It runs classical and graph-based deep detectors behind one interface, selects alarm thresholds without labels, and distills a trained detector down to Power of Ten conformant C that runs on flight hardware. Per-channel SHAP attribution and LLM-generated diagnostic reports are on the roadmap.

Benchmarked on SMAP (NASA). MSL (NASA) and ESA-ADB (ESA) results land in v0.3.0.

---

<table>
  <tr>
    <td align="center"><a href="https://julioanzaldo.github.io/telemetry-anomdet/getting_started.html"><b>Getting Started</b></a><br>Installation and setup</td>
    <td align="center"><a href="https://julioanzaldo.github.io/telemetry-anomdet/api/models/base.html"><b>API Reference</b></a><br>BaseDetector, models, ensemble</td>
    <td align="center"><a href="https://julioanzaldo.github.io/telemetry-anomdet/tutorials/real_time_example.html"><b>Tutorial</b></a><br>End-to-end detection example</td>
    <td align="center"><a href="https://julioanzaldo.github.io/telemetry-anomdet/user_guide/pipeline_overview.html"><b>Pipeline Overview</b></a><br>Ingest, detect, explain</td>
    <td align="center"><a href="https://github.com/JulioAnzaldo/telemetry-anomdet/discussions"><b>Discussions</b></a><br>Questions and feedback</td>
  </tr>
</table>

---

## Quick Install

```bash
# recommended
uv add telemetry-anomdet

# or with pip
pip install telemetry-anomdet
```

The base install is torch-free. The deep detectors (`GDN`, `KANGDN`) need the
`deep` extra:

```bash
uv add "telemetry-anomdet[deep]"
```

## Example Usage

### Fit the Ensemble on Nominal Telemetry

All detectors are trained on **nominal data only**, no anomaly labels used during training. Labels are used exclusively for evaluation.

```python
import numpy as np
from telemetry_anomdet.models.unsupervised.pca import PCAAnomaly
from telemetry_anomdet.models.unsupervised.kmeans import KMeansAnomaly
from telemetry_anomdet.models.ensemble import AnomalyEnsemble

# X_train: (n_windows, window_size, n_features), produced by windowify()
X_train = ...

ensemble = AnomalyEnsemble(
    models = {
        "pca":    PCAAnomaly(n_components = 10),
        "kmeans": KMeansAnomaly(n_clusters = 5, scale = True),
    },
    combine = "mean",
    normalize = "robust",
    percentile = 95.0,
)

ensemble.fit(X_train)
```

### Inference and Anomaly Flags

```python
scores = ensemble.decision_function(X_test)   # higher = more anomalous
flags  = ensemble.is_anomaly(X_test)          # boolean mask at training threshold

# Runtime sensitivity override, no retraining needed
flags_strict = ensemble.is_anomaly(X_test, percentile = 98.0)
flags_custom  = ensemble.is_anomaly(X_test, threshold = 0.75)
```

### Per-Model Score Decomposition

`score_components()` returns raw per-model scores before normalization or combination. This is the SHAP hook: perturb input channels, measure how each sensor drives each model's score independently.

```python
components = ensemble.score_components(X_test)
# {"pca": array(100,), "kmeans": array(100,)}
```

### Getting the SMAP dataset

The benchmarks and examples need a local copy of the NASA SMAP/MSL data, which
is not redistributed with the toolkit. From the root of the repo:

```bash
pip install kaggle

# Requires a Kaggle API key in ~/.kaggle/kaggle.json
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl \
  && mv nasa-anomaly-detection-dataset-smap-msl.zip data.zip \
  && unzip -o data.zip \
  && rm data.zip \
  && mv data/data tmp && rm -r data && mv tmp data
```

That leaves `data/train/`, `data/test/` and `labeled_anomalies.csv`. Point the
examples at it:

```bash
export TAD_SMAP_DIR=data          # Windows: $env:TAD_SMAP_DIR = "data"
python examples/smap_demo.py
python examples/smap_benchmark.py
```

Datasets are never committed; `data/` is gitignored.

### Coming in Phase 3+

```python
# Per-channel SHAP attribution (Phase 3)
# shap_values = explainer.explain(ensemble, X_anomalous)
# {"sensor_01": 0.42, "sensor_02": 0.18, ...}

# LLM diagnostic report (Phase 4)
# report = llm_engine.explain(shap_values, context, channel_names)
# DiagnosticReport(
#     anomaly_type = "power subsystem fault",
#     severity = "high",
#     primary_channels = ["sensor_01", "sensor_07"],
#     explanation = "...",
#     recommended_action = "...",
#     confidence = 0.87,
# )
```

## Features

**Available now**

- **One interface for every detector.** Classical and deep detectors share the same `fit` / `decision_function` / `predict` / `is_anomaly` API, so stacking or swapping models needs no per-model glue.
- **Graph deviation detectors.** `GDN` and `KANGDN` forecast each channel from its learned top-k neighbours and score the deviation, catching readings that are individually plausible but wrong in relation to each other.
- **Alarms without labels.** `threshold_for_budget` and `dynamic_threshold` pick an operating point from the score distribution alone. Operations can state an alarm rate; they cannot state a recall they have no way to observe.
- **Honest evaluation.** Event-level scoring alongside the conventional point-adjusted F1, plus a random baseline row in the benchmark, because point-adjusted F1 rates uniform random noise above every trained configuration on SMAP.
- **Runs on flight hardware.** A fitted `KANGDN` distills to a torch-free NumPy evaluator, then to Power of Ten conformant C with golden vectors. Host and ESP32-S3 targets are in [`targets/`](targets/).
- **Stacking ensemble** with robust (median + IQR) score normalization, built for anomaly scores that are extreme by definition.
- **Per-model score decomposition** via `score_components()`, the hook that enables per-channel attribution without retraining.
- **Spacecraft-native ingestion.** SMAP and CSV load straight into a long-form `TelemetryDataset`, no schema wrangling.
- **Runtime sensitivity control.** `is_anomaly()` accepts a percentile or threshold override, so operators re-tune without retraining.

**On the roadmap**

- `TranAD`: transformer-based sequence reconstruction
- `SHAPExplainer`: per-channel attribution over `score_components()` (Phase 3)
- LLM diagnostic reports, SHAP chart supplied as an image (Phase 4)
- Human-in-the-loop threshold feedback (Phase 5)
- MSL results, using the existing SMAP loader (`spacecraft = "MSL"`) (v0.3.0)
- ESA-ADB evaluation: the multivariate benchmark that exercises the graph (v0.3.0)

## AI Transparency

AI assistants are used while writing and maintaining this project's documentation: drafting API reference pages, tightening prose, and catching claims that go stale.

What that does not change:

- **Numbers come from runs.** Benchmark figures are produced by [`examples/smap_benchmark.py`](examples/smap_benchmark.py), and the hardware results in the deployment docs are measured on an ESP32-S3 by the conformance harness in [`targets/`](targets/). Nothing quoted in these docs is estimated or inferred.
- **Behavior claims are backed by tests.** The suite gates CI, and the documented API is covered by it.

Everything published here is reviewed before it lands.

## Getting Help

### Discussions
To suggest improvements, or ask for help, please see [GitHub Discussions](https://github.com/JulioAnzaldo/telemetry-anomdet/discussions)

### Bug reports
To submit a report on any bugs or issues, [open an issue here](https://github.com/JulioAnzaldo/telemetry-anomdet/issues)
