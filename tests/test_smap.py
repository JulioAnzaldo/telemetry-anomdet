# tests/test_smap.py

"""Unit tests for the SMAP loader, using synthetic .npy fixtures."""

import numpy as np
import pandas as pd
import pytest

from telemetry_anomdet.ingest import (
    TelemetryDataset,
    anomaly_point_mask,
    load_smap,
    load_smap_channel,
    load_smap_labels,
)


def _write_channel(dir_path, chan_id, arr):
    """Save a channel array as <dir_path>/<chan_id>.npy and return the path."""
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{chan_id}.npy"
    np.save(path, arr)
    return path


# --------------------------------------------------------------------------- #
# load_smap_channel
# --------------------------------------------------------------------------- #


def test_load_channel_long_form_shape_and_columns(tmp_path):
    # 4 timesteps, col0 telemetry, col1 active command, col2 all-zero
    arr = np.array([[1.0, 1.0, 0.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0], [4.0, 0.0, 0.0]])
    path = _write_channel(tmp_path / "test", "A-1", arr)

    ds = load_smap_channel(path, "A-1")
    assert isinstance(ds, TelemetryDataset)
    df = ds.to_pandas()

    assert list(df.columns) == ["timestamp", "variable", "value"]
    # nonzero dims are col0 and col1 -> 2 variables x 4 timesteps = 8 rows
    assert len(df) == 8
    assert set(df["variable"].unique()) == {"A-1_dim0", "A-1_dim1"}
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_load_channel_dims_options(tmp_path):
    arr = np.array([[1.0, 5.0, 0.0], [2.0, 6.0, 0.0]])
    path = _write_channel(tmp_path / "test", "B-1", arr)

    # telemetry -> only dim0
    tel = load_smap_channel(path, "B-1", dims="telemetry").to_pandas()
    assert set(tel["variable"].unique()) == {"B-1_dim0"}

    # all -> every column including the all-zero one
    alld = load_smap_channel(path, "B-1", dims="all").to_pandas()
    assert set(alld["variable"].unique()) == {"B-1_dim0", "B-1_dim1", "B-1_dim2"}

    # explicit list
    expl = load_smap_channel(path, "B-1", dims=[1]).to_pandas()
    assert set(expl["variable"].unique()) == {"B-1_dim1"}
    assert expl["value"].tolist() == [5.0, 6.0]


def test_load_channel_values_and_cadence(tmp_path):
    arr = np.array([[10.0], [11.0], [12.0]])
    path = _write_channel(tmp_path / "test", "C-1", arr)

    df = load_smap_channel(path, "C-1", cadence="5s", start="2001-01-01").to_pandas()
    assert df["value"].tolist() == [10.0, 11.0, 12.0]
    deltas = df["timestamp"].diff().dropna().dt.total_seconds().unique()
    assert list(deltas) == [5.0]


def test_load_channel_infers_id_from_filename(tmp_path):
    path = _write_channel(tmp_path / "test", "D-2", np.array([[1.0], [2.0]]))
    df = load_smap_channel(path).to_pandas()
    assert df["variable"].str.startswith("D-2").all()


def test_load_channel_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_smap_channel(tmp_path / "test" / "nope.npy")


def test_load_channel_bad_dims_index_raises(tmp_path):
    path = _write_channel(tmp_path / "test", "E-1", np.array([[1.0], [2.0]]))
    with pytest.raises(ValueError):
        load_smap_channel(path, dims=[5])


# --------------------------------------------------------------------------- #
# load_smap (multi-channel)
# --------------------------------------------------------------------------- #


def test_load_smap_combines_channels(tmp_path):
    _write_channel(tmp_path / "test", "A-1", np.array([[1.0], [2.0]]))
    _write_channel(tmp_path / "test", "A-2", np.array([[3.0], [4.0]]))

    ds = load_smap(tmp_path, ["A-1", "A-2"], split="test")
    df = ds.to_pandas()
    assert set(df["variable"].unique()) == {"A-1_dim0", "A-2_dim0"}
    assert len(df) == 4


def test_load_smap_requires_channels(tmp_path):
    with pytest.raises(ValueError):
        load_smap(tmp_path, [], split="test")


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


def test_load_smap_labels_parses_and_filters(tmp_path):
    csv = tmp_path / "labeled_anomalies.csv"
    pd.DataFrame(
        {
            "chan_id": ["A-1", "M-1"],
            "spacecraft": ["SMAP", "MSL"],
            "anomaly_sequences": ["[[0, 2], [5, 5]]", "[[1, 1]]"],
            "num_values": [100, 100],
        }
    ).to_csv(csv, index=False)

    labels = load_smap_labels(csv, spacecraft="SMAP")
    assert len(labels) == 1
    assert labels.loc[0, "chan_id"] == "A-1"
    assert labels.loc[0, "sequences"] == [(0, 2), (5, 5)]
    assert labels.loc[0, "anomaly_span"] == 4  # (2-0+1) + (5-5+1)


def test_load_smap_labels_unknown_spacecraft_raises(tmp_path):
    csv = tmp_path / "labeled_anomalies.csv"
    pd.DataFrame(
        {
            "chan_id": ["A-1"],
            "spacecraft": ["SMAP"],
            "anomaly_sequences": ["[[0, 1]]"],
        }
    ).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        load_smap_labels(csv, spacecraft="MSL")


def test_anomaly_point_mask():
    mask = anomaly_point_mask([(1, 3), (6, 6)], n_timesteps=8)
    expected = [False, True, True, True, False, False, True, False]
    assert mask.tolist() == expected


def test_anomaly_point_mask_clips_out_of_range():
    mask = anomaly_point_mask([(-2, 1), (5, 100)], n_timesteps=6)
    assert mask.tolist() == [True, True, False, False, False, True]
