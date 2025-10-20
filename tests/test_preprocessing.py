#tests/test_preprocessing.py

"""
Skeleton for pre processing unit testing.
"""

import importlib

def test_preprocessing_module_import():
    mod = importlib.import_module("telemetry_anomdet.preprocessing.preprocessing")
    assert mod is not None

def test_clean_exists_and_is_callable():
    """Ensure clean() exists and is callable."""
    from telemetry_anomdet import preprocessing
    assert hasattr(preprocessing, "clean")
    assert callable(preprocessing.clean)