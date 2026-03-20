<h2 align="center">telemetry-anomdet</h2>

<p align="center">
  <a href="https://julioanzaldo.github.io/telemetry-anomdet/"><img alt="Documentation" src="https://img.shields.io/badge/docs-online-blue"></a>
  <a href="https://test.pypi.org/project/telemetry-anomdet/"><img alt="Test PyPI" src="https://img.shields.io/badge/Test-PyPI-yellow?logo=pypi&logoColor=white"></a>
  <a href="https://github.com/JulioAnzaldo/telemetry-anomdet/issues"><img alt="GitHub Issues" src="https://img.shields.io/github/issues/JulioAnzaldo/telemetry-anomdet"></a>
  <a href="https://github.com/JulioAnzaldo/telemetry-anomdet/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/JulioAnzaldo/telemetry-anomdet"></a>
</p>

**telemetry-anomdet** is an open-source anomaly detection toolkit for spacecraft telemetry. It runs a stacking ensemble of classical and deep learning detectors with per-channel SHAP attribution, SymTorch symbolic fault expressions, and LLM-generated diagnostic reports.

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
pip install telemetry-anomdet
```

## Example Usage

### Fit the Ensemble on Nominal Telemetry

All detectors are trained on **nominal data only** -- no anomaly labels used during training. Labels are used exclusively for evaluation.

```python
import numpy as np
from telemetry_anomdet.models.unsupervised.pca import PCAAnomaly
from telemetry_anomdet.models.unsupervised.kmeans import KMeansAnomaly
from telemetry_anomdet.models.ensemble import AnomalyEnsemble

# X_train: (n_windows, window_size, n_features) -- produced by windowify()
X_train = ...

ensemble = AnomalyEnsemble(
    models={
        "pca":    PCAAnomaly(n_components=10),
        "kmeans": KMeansAnomaly(n_clusters=5, scale=True),
    },
    combine="mean",
    normalize="robust",
    percentile=95.0,
)

ensemble.fit(X_train)
```

### Inference and Anomaly Flags

```python
scores = ensemble.decision_function(X_test)   # higher = more anomalous
flags  = ensemble.is_anomaly(X_test)          # boolean mask at training threshold

# Runtime sensitivity override -- no retraining needed
flags_strict = ensemble.is_anomaly(X_test, percentile=98.0)
flags_custom  = ensemble.is_anomaly(X_test, threshold=0.75)
```

### Per-Model Score Decomposition

`score_components()` returns raw per-model scores before normalization or combination. This is the SHAP hook: perturb input channels, measure how each sensor drives each model's score independently.

```python
components = ensemble.score_components(X_test)
# {"pca": array(100,), "kmeans": array(100,)}
```

### Coming in Phase 3+

```python
# Per-channel SHAP attribution (Phase 3)
# shap_values = explainer.explain(ensemble, X_anomalous)
# {"sensor_01": 0.42, "sensor_02": 0.18, ...}

# LLM diagnostic report (Phase 4)
# report = llm_engine.explain(shap_values, context, channel_names)
# DiagnosticReport(
#     anomaly_type="power subsystem fault",
#     severity="high",
#     primary_channels=["sensor_01", "sensor_07"],
#     explanation="...",
#     recommended_action="...",
#     confidence=0.87,
# )
```

## Features

| Feature | Status |
|---|---|
| Unified `BaseDetector` interface -- all detectors share `fit` / `decision_function` / `predict` / `is_anomaly`, classical and deep | Complete |
| Stacking ensemble with robust normalization (median + IQR) -- anomaly scores are extreme by definition, minmax would compress the normal range | Complete |
| `score_components()` SHAP hook -- per-model raw scores before combination, enabling per-channel attribution without retraining | Complete |
| `GDN` -- graph deviation network that learns inter-sensor relationships; detects relational faults invisible to univariate methods | Phase 2 |
| `TranAD` -- transformer sequence reconstruction; consumes raw 3D windows, no flattening | Phase 2 |
| Per-channel SHAP attribution over `score_components()` | Phase 3 |
| LLM diagnostic reports on Jetson Orin -- Llama 3.1 8B via llama.cpp, SHAP chart as image input per ICLR 2025 findings | Phase 4 |
| Human-in-the-loop feedback -- operator flags adjust `is_anomaly()` threshold dynamically, no retraining | Phase 5 |
| Cross-dataset generalization -- ensemble trained on SMAP evaluated on OPS-SAT without retraining | Phase 6 |
| SymTorch symbolic distillation of GDN -- extracts human-readable fault expressions per sensor relationship, giving the LLM mechanism-level context beyond SHAP weights | Stretch |

## Getting Help

### Discussions
To suggest improvements, or ask for help, please see [GitHub Discussions](https://github.com/JulioAnzaldo/telemetry-anomdet/discussions)

### Bug reports
To submit a report on any bugs or issues, [open an issue here](https://github.com/JulioAnzaldo/telemetry-anomdet/issues)
