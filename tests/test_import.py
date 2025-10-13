# tests/test_import.py

"""
Basic smoke tests for the telemetry_anomdet package.

These tests verify that the package and its core
modules can be imported successfully, and that a
__version__ attribute is defined.
"""

import importlib
import telemetry_anomdet

def test_package_import():
    """Package imports without error."""
    package = importlib.import_module("telemetry_anomdet")
    assert package is not None

def test_version_present():
    """Package exposes non-empty __version__ string."""
    assert hasattr(telemetry_anomdet, "__version__")
    v = telemetry_anomdet.__version__
    assert isinstance(v, str) and v.strip() != ""

def test_core_modules_import():
    """Core submodules import without error."""
    modules = {
        "telemetry_anomdet.preprocessing",
        "telemetry_anomdet.features",
        "telemetry_anomdet.models",
    }

    for m in modules:
        mod = importlib.import_module(m)
        assert mod is not None