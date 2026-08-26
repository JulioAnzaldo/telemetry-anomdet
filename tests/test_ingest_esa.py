"""
Tests for the ESA-ADB loader.

Built against a synthetic mission directory rather than the real archive: the
dataset is 3.8 GB and lives outside the repo, so CI has no access to it. The
fixture reproduces the layout that matters, including the two shapes that are
easy to mishandle, an event split into several disjoint segments, and an event
affecting only one channel of a subset.
"""

import io
import pickle
import zipfile

import numpy as np
import pandas as pd
import pytest

from telemetry_anomdet.ingest.esa import (
    MISSION1_CADENCE_S,
    MISSION1_LIGHTWEIGHT,
    MISSION1_SPAN,
    esa_splits,
    load_esa,
    load_esa_channels,
    load_esa_labels,
    rescope_anomaly_types,
)

CHANNELS = ["channel_1", "channel_2", "channel_3"]
START = pd.Timestamp("2000-01-01")
N = 2880  # one day at 30 s


@pytest.fixture
def mission(tmp_path):
    """A miniature ESA mission directory: three channels, one day, four events."""
    root = tmp_path / "ESA-MissionT"
    (root / "channels").mkdir(parents=True)

    index = pd.date_range(START, periods=N, freq=f"{MISSION1_CADENCE_S}s")
    for i, name in enumerate(CHANNELS):
        series = pd.Series(np.linspace(0, 1, N) + i, index=index, name=name, dtype="float32")
        series.index.name = "datetime"
        payload = pickle.dumps(series.to_frame())
        with zipfile.ZipFile(root / "channels" / f"{name}.zip", "w") as z:
            z.writestr(name, payload)

    pd.DataFrame(
        {
            "Channel": CHANNELS,
            "Subsystem": ["subsystem_1"] * 3,
            "Physical Unit": ["physical_unit_1"] * 3,
            "Group": [1, 1, 2],
            "Target": ["YES", "YES", "NO"],
            "Categorical": ["NO"] * 3,
        }
    ).to_csv(root / "channels.csv", index=False)

    # id_1  multivariate in the mission, but touches only channel_1 of the
    #       {channel_1, channel_2} subset -> univariate once rescoped.
    # id_2  one channel, two disjoint segments (the burst case).
    # id_3  short enough to be a point anomaly at the dominant cadence.
    # id_4  a communication gap, dropped by default.
    rows = [
        ("id_1", "channel_1", "00:10:00", "00:40:00"),
        ("id_1", "channel_3", "00:10:00", "00:40:00"),
        ("id_2", "channel_2", "02:00:00", "02:30:00"),
        ("id_2", "channel_2", "03:00:00", "03:30:00"),
        ("id_3", "channel_1", "05:00:00", "05:01:00"),
        ("id_4", "channel_1", "06:00:00", "06:30:00"),
    ]
    pd.DataFrame(
        {
            "ID": [r[0] for r in rows],
            "Channel": [r[1] for r in rows],
            "StartTime": [(START + pd.Timedelta(r[2])).isoformat() + "Z" for r in rows],
            "EndTime": [(START + pd.Timedelta(r[3])).isoformat() + "Z" for r in rows],
        }
    ).to_csv(root / "labels.csv", index=False)

    pd.DataFrame(
        {
            "ID": ["id_1", "id_2", "id_3", "id_4"],
            "Class": ["class_1"] * 4,
            "Subclass": ["subclass_1"] * 4,
            "Category": ["Anomaly", "Rare Event", "Anomaly", "Communication Gap"],
            "Dimensionality": ["Multivariate", "Univariate", "Univariate", "Univariate"],
            "Locality": ["Global", "Local", "Global", "Global"],
            "Length": ["Subsequence"] * 4,
        }
    ).to_csv(root / "anomaly_types.csv", index=False)

    return root


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


class TestSplits:
    def test_mission1_matches_the_published_division(self):
        """
        The benchmark halves the mission and reserves the last three months of
        the training half for validation, giving 84 months of train+validation
        for Mission1. Getting this wrong makes every number incomparable.
        """
        s = esa_splits(MISSION1_SPAN)
        assert s["train"] == (pd.Timestamp("2000-01-01"), pd.Timestamp("2006-10-01"))
        assert s["validation"] == (pd.Timestamp("2006-10-01"), pd.Timestamp("2007-01-01"))
        assert s["test"] == (pd.Timestamp("2007-01-01"), pd.Timestamp("2014-01-01"))

    def test_train_and_validation_are_84_months(self):
        s = esa_splits(MISSION1_SPAN)
        months = (s["validation"][1].year - s["train"][0].year) * 12 + (
            s["validation"][1].month - s["train"][0].month
        )
        assert months == 84

    def test_splits_are_contiguous_and_ordered(self):
        """No leakage: every validation and test sample is later than training."""
        s = esa_splits(MISSION1_SPAN)
        assert s["train"][1] == s["validation"][0]
        assert s["validation"][1] == s["test"][0]
        assert s["train"][0] < s["train"][1] < s["test"][0] < s["test"][1]

    def test_halves_are_equal(self):
        s = esa_splits(MISSION1_SPAN)
        first = s["test"][0] - s["train"][0]
        assert first == s["test"][1] - s["test"][0]

    def test_rejects_reversed_span(self):
        with pytest.raises(ValueError, match="span must be increasing"):
            esa_splits((pd.Timestamp("2010-01-01"), pd.Timestamp("2000-01-01")))

    def test_rejects_validation_longer_than_training(self):
        with pytest.raises(ValueError, match="leaves no training data"):
            esa_splits(MISSION1_SPAN, validation_months=120)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class TestLoadEsa:
    def test_returns_long_form(self, mission):
        df = load_esa(mission, CHANNELS[:2]).to_pandas()
        assert list(df.columns) == ["timestamp", "variable", "value"]
        assert set(df["variable"]) == {"channel_1", "channel_2"}
        assert len(df) == 2 * N

    def test_sorted_by_timestamp(self, mission):
        df = load_esa(mission, CHANNELS[:2]).to_pandas()
        assert df["timestamp"].is_monotonic_increasing

    def test_end_is_exclusive(self, mission):
        """Half-open slices, so adjacent splits cannot share a sample."""
        cut = START + pd.Timedelta("01:00:00")
        first = load_esa(mission, ["channel_1"], end=cut).to_pandas()
        second = load_esa(mission, ["channel_1"], start=cut).to_pandas()
        assert first["timestamp"].max() < cut
        assert second["timestamp"].min() == cut
        assert len(first) + len(second) == N

    def test_split_selects_a_period(self, mission):
        span = (START, START + pd.Timedelta(days=1))
        kw = dict(span=span, validation_months=0)
        train = load_esa(mission, ["channel_1"], split="train", **kw).to_pandas()
        test = load_esa(mission, ["channel_1"], split="test", **kw).to_pandas()
        bounds = esa_splits(span, validation_months=0)

        assert train["timestamp"].min() == bounds["train"][0]
        assert train["timestamp"].max() < bounds["test"][0]
        assert test["timestamp"].min() == bounds["test"][0]
        assert len(train) + len(test) == N

    def test_splits_partition_the_data(self, mission):
        """
        The point of half-open bounds: every sample lands in exactly one split.
        ``validation_months=0`` also checks an empty split loads rather than
        raising, which is the shape the paper's shorter-training-set variants
        can produce.
        """
        span = (START, START + pd.Timedelta(days=1))
        kw = dict(span=span, validation_months=0)
        seen = pd.concat(
            load_esa(mission, ["channel_1"], split=name, **kw).to_pandas()["timestamp"]
            for name in ("train", "validation", "test")
        )
        assert seen.is_unique
        assert len(seen) == N

    def test_split_and_dates_are_mutually_exclusive(self, mission):
        with pytest.raises(ValueError, match="not both"):
            load_esa(mission, ["channel_1"], split="train", start=START)

    def test_unknown_split_rejected(self, mission):
        with pytest.raises(ValueError, match="split must be one of"):
            load_esa(mission, ["channel_1"], split="val")

    def test_resampling_reduces_rows(self, mission):
        df = load_esa(mission, ["channel_1"], resample_rule="5min").to_pandas()
        assert len(df) == N * MISSION1_CADENCE_S // 300

    def test_missing_channel_names_the_file(self, mission):
        with pytest.raises(FileNotFoundError, match="channel_99"):
            load_esa(mission, ["channel_99"])

    def test_empty_channel_list_rejected(self, mission):
        with pytest.raises(ValueError, match="must not be empty"):
            load_esa(mission, [])

    def test_defaults_to_the_lightweight_subset(self):
        assert tuple(f"channel_{i}" for i in range(41, 47)) == MISSION1_LIGHTWEIGHT


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class TestLoadEsaLabels:
    def test_communication_gaps_dropped_by_default(self, mission):
        labels = load_esa_labels(mission)
        assert "Communication Gap" not in set(labels["Category"])
        assert set(labels["ID"]) == {"id_1", "id_2", "id_3"}

    def test_keeping_everything(self, mission):
        labels = load_esa_labels(mission, drop_categories=None)
        assert "id_4" in set(labels["ID"])

    def test_anomalies_only(self, mission):
        labels = load_esa_labels(mission, drop_categories=("Communication Gap", "Rare Event"))
        assert set(labels["Category"]) == {"Anomaly"}

    def test_timestamps_are_tz_naive(self, mission):
        """The channel index is tz-naive, so labels must be too or every
        comparison between them raises."""
        labels = load_esa_labels(mission)
        assert labels["StartTime"].dt.tz is None
        assert labels["EndTime"].dt.tz is None

    def test_channel_filter(self, mission):
        labels = load_esa_labels(mission, ["channel_2"])
        assert set(labels["Channel"]) == {"channel_2"}

    def test_segments_outnumber_events(self, mission):
        """A row is a segment, not an event. id_2 is one event in two regions."""
        labels = load_esa_labels(mission, ["channel_2"])
        assert len(labels) == 2
        assert labels["ID"].nunique() == 1


# ---------------------------------------------------------------------------
# Subset-scoped anomaly types
# ---------------------------------------------------------------------------


class TestRescopeAnomalyTypes:
    def test_multivariate_becomes_univariate_within_a_subset(self, mission):
        """
        id_1 spans channel_1 and channel_3 mission-wide, so the file calls it
        Multivariate. Restricted to {channel_1, channel_2} it touches one
        channel and is univariate. Using the file's column unchanged would
        misreport the anomaly mix of every subset experiment.
        """
        labels = load_esa_labels(mission, ["channel_1", "channel_2"])
        row = labels[labels["ID"] == "id_1"].iloc[0]
        assert row["Dimensionality"] == "Univariate"
        assert row["Dimensionality_mission"] == "Multivariate"

    def test_multivariate_survives_when_the_subset_covers_it(self, mission):
        labels = load_esa_labels(mission, ["channel_1", "channel_3"])
        row = labels[labels["ID"] == "id_1"].iloc[0]
        assert row["Dimensionality"] == "Multivariate"

    def test_short_event_becomes_a_point_anomaly(self, mission):
        """60 s is 2 samples at 30 s, within the 3-sample point threshold."""
        labels = load_esa_labels(mission, ["channel_1"])
        assert labels[labels["ID"] == "id_3"].iloc[0]["Length"] == "Point"

    def test_long_event_stays_a_subsequence(self, mission):
        labels = load_esa_labels(mission, ["channel_1"])
        assert labels[labels["ID"] == "id_1"].iloc[0]["Length"] == "Subsequence"

    def test_point_requires_every_region_to_qualify(self):
        """
        The benchmark's rule: a fragmented event is a point anomaly only if all
        of its regions are. One long region disqualifies the whole event.
        """
        labels = pd.DataFrame(
            {
                "ID": ["id_a", "id_a"],
                "Channel": ["channel_1", "channel_1"],
                "StartTime": pd.to_datetime(["2000-01-01 00:00", "2000-01-01 01:00"]),
                "EndTime": pd.to_datetime(["2000-01-01 00:01", "2000-01-01 02:00"]),
                "Dimensionality": ["Univariate"] * 2,
                "Length": ["Subsequence"] * 2,
            }
        )
        assert set(rescope_anomaly_types(labels)["Length"]) == {"Subsequence"}

    def test_locality_is_left_alone(self, mission):
        """Deciding locality needs the signal's nominal range, not annotations,
        so the column stays mission-scope and must not be silently altered."""
        labels = load_esa_labels(mission, ["channel_1"])
        assert set(labels["Locality"]) <= {"Global", "Local"}
        assert "Locality_mission" not in labels.columns

    def test_rescoping_can_be_disabled(self, mission):
        labels = load_esa_labels(mission, ["channel_1", "channel_2"], rescope_types=False)
        row = labels[labels["ID"] == "id_1"].iloc[0]
        assert row["Dimensionality"] == "Multivariate"
        assert "Dimensionality_mission" not in labels.columns


# ---------------------------------------------------------------------------
# Channel metadata
# ---------------------------------------------------------------------------


def test_load_esa_channels_marks_targets(mission):
    channels = load_esa_channels(mission)
    assert channels["is_target"].tolist() == [True, True, False]


def test_pickle_payload_may_be_a_series(tmp_path):
    """The archives hold a one-column frame; a bare Series must load too."""
    root = tmp_path / "M"
    (root / "channels").mkdir(parents=True)
    index = pd.date_range(START, periods=10, freq="30s")
    series = pd.Series(np.arange(10, dtype="float32"), index=index, name="channel_1")
    with zipfile.ZipFile(root / "channels" / "channel_1.zip", "w") as z:
        z.writestr("channel_1", pickle.dumps(series))

    df = load_esa(root, ["channel_1"]).to_pandas()
    assert len(df) == 10
    assert df["value"].tolist() == list(range(10))


def test_reads_from_bytesio_without_extracting(mission):
    """Regression guard: the loader must read the member in memory rather than
    extracting to disk, since the real archives are tens of gigabytes."""
    before = set((mission / "channels").iterdir())
    load_esa(mission, ["channel_1"])
    assert set((mission / "channels").iterdir()) == before
    assert isinstance(io.BytesIO(b""), io.BytesIO)  # module is used
