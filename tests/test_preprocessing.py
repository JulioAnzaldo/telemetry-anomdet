# tests/test_preprocessing.py

"""Unit tests for the preprocessing pipeline."""

import importlib

import numpy as np
import pandas as pd
import pytest

from telemetry_anomdet.preprocessing import preprocessing as pp


def _long(rows):
    """Build a long-form telemetry frame from (timestamp, variable, value) tuples."""
    return pd.DataFrame(rows, columns=["timestamp", "variable", "value"])


def _utc(seconds):
    """A UTC timestamp `seconds` past a fixed epoch."""
    return pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(seconds=seconds)


# --------------------------------------------------------------------------- #
# module / API surface
# --------------------------------------------------------------------------- #


def test_preprocessing_module_import():
    mod = importlib.import_module("telemetry_anomdet.preprocessing.preprocessing")
    assert mod is not None


def test_public_api_is_callable():
    from telemetry_anomdet import preprocessing

    for name in (
        "clean",
        "dedupe",
        "integrity_check",
        "resample",
        "interpolate_gaps",
        "normalize_fit",
        "pipeline",
    ):
        assert callable(getattr(preprocessing, name))


# --------------------------------------------------------------------------- #
# clean
# --------------------------------------------------------------------------- #


def test_clean_drops_nulls_and_out_of_bounds():
    df = _long(
        [
            (_utc(0), "temp", 10.0),
            (_utc(1), "temp", np.nan),  # dropped: null value
            (_utc(2), "temp", 999.0),  # dropped: above physical bound
            (_utc(3), "volt", 5.0),
        ]
    )
    out = pp.clean(df, physical_bounds={"temp": (0, 100)})

    assert len(out) == 2
    assert out["value"].notna().all()
    assert 999.0 not in out["value"].values


def test_clean_bounds_support_glob_patterns():
    df = _long(
        [
            (_utc(0), "Battery_Temp", -100.0),  # dropped by pattern bound
            (_utc(1), "Battery_Temp", 20.0),
        ]
    )
    out = pp.clean(df, physical_bounds={"Battery_*": (-40, 85)})
    assert out["value"].tolist() == [20.0]


def test_clean_rejects_malformed_bounds():
    df = _long([(_utc(0), "x", 1.0)])
    with pytest.raises(ValueError):
        pp.clean(df, physical_bounds={"x": (1, 2, 3)})


# --------------------------------------------------------------------------- #
# dedupe
# --------------------------------------------------------------------------- #


def test_dedupe_removes_exact_and_retransmits():
    df = _long(
        [
            (_utc(0), "a", 1.0),
            (_utc(0), "a", 1.0),  # exact duplicate
            (_utc(0), "a", 2.0),  # retransmit, same (ts, var) -> keep last
            (_utc(1), "a", 3.0),
        ]
    )
    out = pp.dedupe(df)

    assert len(out) == 2
    # keep="last" after sorting by value -> the 2.0 reading wins at t0
    assert out.loc[out["timestamp"] == _utc(0), "value"].iloc[0] == 2.0


def test_dedupe_rejects_non_dataframe():
    with pytest.raises(TypeError):
        pp.dedupe([1, 2, 3])


# --------------------------------------------------------------------------- #
# integrity_check
# --------------------------------------------------------------------------- #


def test_integrity_check_passes_valid_frame():
    df = _long([(_utc(0), "a", 1.0), (_utc(1), "a", 2.0)])
    assert pp.integrity_check(df) is None


def test_integrity_check_requires_utc():
    naive = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "variable": ["a", "a"],
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError):
        pp.integrity_check(naive, require_utc=True)


def test_integrity_check_rejects_non_numeric_value():
    df = _long([(_utc(0), "a", "oops")])
    with pytest.raises(ValueError):
        pp.integrity_check(df, require_utc=False, require_sorted=False)


def test_integrity_check_rejects_unsorted():
    df = _long([(_utc(5), "a", 1.0), (_utc(0), "a", 2.0)])
    with pytest.raises(ValueError):
        pp.integrity_check(df, require_utc=True, require_sorted=True)


# --------------------------------------------------------------------------- #
# resample
# --------------------------------------------------------------------------- #


def test_resample_regularizes_to_uniform_cadence():
    df = _long(
        [
            (_utc(0), "a", 1.0),
            (_utc(3), "a", 3.0),
            (_utc(7), "a", 7.0),
        ]
    )
    out = pp.resample(df, rule="5s", agg="mean")

    assert list(out.columns) == ["timestamp", "variable", "value"]
    assert out["value"].notna().all()
    assert out["timestamp"].is_monotonic_increasing


def test_resample_rejects_unknown_agg():
    df = _long([(_utc(0), "a", 1.0)])
    with pytest.raises(ValueError):
        pp.resample(df, rule="5s", agg="bogus")


# --------------------------------------------------------------------------- #
# interpolate_gaps
# --------------------------------------------------------------------------- #


def test_interpolate_gaps_ffill_fills_within_limit():
    # 'a' missing at t1 (only 'b' reports there) -> NaN in wide form
    df = _long(
        [
            (_utc(0), "a", 1.0),
            (_utc(1), "b", 2.0),
            (_utc(2), "a", 3.0),
        ]
    )
    out = pp.interpolate_gaps(df, method="ffill", limit=1)

    a = out[out["variable"] == "a"].sort_values("timestamp")
    assert a["value"].tolist() == [1.0, 1.0, 3.0]  # gap forward-filled


def test_interpolate_gaps_linear_interpolates():
    df = _long(
        [
            (_utc(0), "a", 1.0),
            (_utc(1), "b", 0.0),
            (_utc(2), "a", 3.0),
        ]
    )
    out = pp.interpolate_gaps(df, method="linear", limit=1)

    a = out[out["variable"] == "a"].sort_values("timestamp")
    assert a["value"].tolist() == [1.0, 2.0, 3.0]  # midpoint interpolated


def test_interpolate_gaps_rejects_non_dataframe():
    with pytest.raises(TypeError):
        pp.interpolate_gaps("not a frame")


# --------------------------------------------------------------------------- #
# normalize_fit
# --------------------------------------------------------------------------- #


def test_normalize_fit_zscore():
    df = _long([(_utc(i), "a", v) for i, v in enumerate([1, 2, 3, 4, 5])])
    params = pp.normalize_fit(df, method="zscore")

    center, scale = params["a"]
    assert center == pytest.approx(3.0)
    assert scale == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))


def test_normalize_fit_minmax():
    df = _long([(_utc(i), "a", v) for i, v in enumerate([1, 2, 3, 4, 5])])
    params = pp.normalize_fit(df, method="minmax")
    assert params["a"] == (1.0, 4.0)  # (min, range)


def test_normalize_fit_constant_variable_guards_zero_scale():
    df = _long([(_utc(i), "c", 7.0) for i in range(4)])
    for method in ("zscore", "minmax"):
        center, scale = pp.normalize_fit(df, method=method)["c"]
        assert scale == 1.0  # never zero -> safe to divide by at inference


def test_normalize_fit_rejects_unknown_method():
    df = _long([(_utc(0), "a", 1.0)])
    with pytest.raises(ValueError):
        pp.normalize_fit(df, method="bogus")


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


def test_pipeline_returns_clean_dataframe():
    df = _long(
        [
            (_utc(0), "a", 1.0),
            (_utc(0), "a", 1.0),  # duplicate
            (_utc(2), "a", np.nan),  # null
            (_utc(4), "a", 5.0),
            (_utc(0), "b", 10.0),
            (_utc(4), "b", 12.0),
        ]
    )
    out = pp.pipeline(df, resample_rule="2s")

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["timestamp", "variable", "value"]
    assert out["value"].notna().all()
    assert out["timestamp"].is_monotonic_increasing


def test_pipeline_skips_resample_when_rule_none():
    df = _long([(_utc(0), "a", 1.0), (_utc(1), "a", 2.0)])
    out = pp.pipeline(df, resample_rule=None)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 2
