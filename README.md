<h2 align="center">telemetry-anomdet</h2>

An open-source anomaly detection toolkit for spacecraft telemetry. Combines classical and deep learning detectors in a stacking ensemble with explainable AI. Designed to deliver actionable diagnostics within the ground station inter-pass window.

[![Documentation](https://img.shields.io/badge/docs-Online-blue)](https://julioanzaldo.github.io/telemetry-anomdet/)
[![Test PyPI](https://img.shields.io/testpypi/v/telemetry-anomdet?label=Test%20PyPI&color=yellow&logo=pypi)](https://test.pypi.org/project/telemetry-anomdet/)

## Overview
telemetry-anomdet simplifies the ground station telemetry pipeline:

- Ingesting heterogeneous telemetry (CCSDS, CSV, HDF5)
- Applying preprocessing (resampling, noise filtering, gap interpolation, physical bounds validation)
- Extracting features (statistical, rolling window, sliding window tensors for sequence models)
- Running a stacking ensemble of unsupervised anomaly detectors with dynamic thresholding
- Explaining detections via per-channel attribution and human-in-the-loop feedback

Validated on SMAP, OPSSAT.

## System requirements

1. MacOS, Linux, or Windows with WSL
2. [git](https://git-scm.com/)
3. [Python 3.9+](https://www.python.org/downloads/), [virtual environments](https://docs.python.org/3/library/venv.html), and [PIP](https://pypi.org/project/pip/)

## Getting Help

### Discussions
To suggest improvements, or ask for help, please see [GitHub Discussions](https://github.com/JulioAnzaldo/telemetry-anomdet/discussions)

### Bug reports
To submit a report on any bugs or issues, [open an issue here](https://github.com/JulioAnzaldo/telemetry-anomdet/issues)

## Getting Started

To get started with telemetry-anomdet, install the toolking with:
```
pip install telemetry-anomdet
```
