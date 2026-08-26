# src/telemetry_anomdet/ingest/esa.py

"""
Loader for the ESA Anomaly Dataset / ESA-ADB (Zenodo DOI 10.5281/zenodo.12528696).

A mission archive holds one ``channels/channel_N.zip`` per telemetry channel and
one ``telecommands/telecommand_N.zip`` per command, alongside four metadata CSVs:

``channels.csv``
    ``Channel, Subsystem, Physical Unit, Group, Target, Categorical``. Only
    ``Target = YES`` channels are monitored for anomalies (58 of 76 in Mission1);
    the rest exist to support detection.
``labels.csv``
    ``ID, Channel, StartTime, EndTime``. One row per *segment*, so a single event
    ``ID`` appears once per affected channel and may appear several times for one
    channel when the event is a burst of separate disturbances.
``anomaly_types.csv``
    ``ID, Class, Subclass, Category, Dimensionality, Locality, Length``, where
    ``Category`` is Anomaly, Rare Event, or Communication Gap.
``telecommands.csv``
    ``Telecommand, Priority``.

Unlike SMAP/MSL this dataset carries real timestamps, so nothing is synthesised.
Output is the canonical long form ``[timestamp, variable, value]`` wrapped in a
:class:`TelemetryDataset`.

.. warning::
   Channel payloads are **pickled pandas objects**, so loading executes the
   pickle stream. That is safe for the official Zenodo archive and is not safe
   for arbitrary user-supplied paths. Point this loader at data you trust.

Two facts about the annotations drive how results must be scored, and both
differ from SMAP:

* An event is identified by its ``ID``, not by contiguity. Roughly a fifth of
  the (event, channel) pairs in the Mission1 lightweight subset are split across
  two to five segments, deliberately, so that a burst of related disturbances
  counts once. Scoring segments as separate events inflates false positives and
  deflates recall at the same time.
* ``Rare Event`` rows are atypical but expected behaviour (commanded manoeuvres,
  resets, calibrations). ESA-ADB's headline results score every category except
  ``Communication Gap``; :func:`load_esa_labels` keeps the column so either
  convention can be applied explicitly rather than by accident.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from telemetry_anomdet.ingest.dataset import TelemetryDataset

# Canonical columns
_TS, _VAR, _VAL = "timestamp", "variable", "value"

#: Mission1's lightweight subset: the six channels ESA-ADB reports its headline
#: benchmark on, all of subsystem_5 / physical_unit_4 / group 8. The benchmark's
#: full 58-channel set defeats every algorithm it published, so this subset is
#: where a comparable result can be had.
MISSION1_LIGHTWEIGHT = tuple(f"channel_{i}" for i in range(41, 47))

#: Categories excluded from ESA-ADB's headline scoring.
NON_EVENT_CATEGORIES = ("Communication Gap",)

#: Mission1's span, end exclusive. Used to derive the benchmark splits without
#: opening a 15-million-row archive first.
MISSION1_SPAN = (pd.Timestamp("2000-01-01"), pd.Timestamp("2014-01-01"))

#: Mission1's dominant sampling frequency is 0.033 Hz. This is the grid the
#: benchmark's own definitions are stated against (a point anomaly is up to
#: three samples *after resampling to the dominant frequency*), so it is the
#: canonical cadence rather than merely a convenient one.
MISSION1_CADENCE_S = 30

#: Longest event, in samples at the dominant cadence, still counted as a point
#: anomaly rather than a subsequence.
POINT_MAX_SAMPLES = 3


def esa_splits(
    span: tuple[pd.Timestamp, pd.Timestamp] = MISSION1_SPAN,
    *,
    validation_months: int = 3,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """
    The benchmark's train / validation / test division.

    ESA-ADB halves each mission chronologically, trains on the first half, and
    reserves the last three months of that half for validation, so every
    validation and test sample is later than every training one.

    Arguments:
        span: ``(start, end)`` of the mission, end exclusive.
        validation_months: Months at the end of the training half held out.
    Returns:
        dict: ``{"train": (start, end), "validation": ..., "test": ...}``, each
            half-open ``[start, end)``.

    Notes:
        Anomalies appear in **all three** splits, training included. That is
        deliberate on the benchmark's part, and it breaks an assumption
        inherited from SMAP, whose training split is anomaly-free: the
        per-channel error median and IQR that calibrate the deviation score are
        fitted on data that contains faults. Both statistics are robust, which
        is what makes the arrangement workable, but a detector calibrated here
        is not calibrated on clean data.
    """
    start, end = pd.Timestamp(span[0]), pd.Timestamp(span[1])
    if end <= start:
        raise ValueError(f"span must be increasing, got {start} to {end}")

    midpoint = start + (end - start) / 2
    validation_start = midpoint - pd.DateOffset(months=validation_months)
    if validation_start <= start:
        raise ValueError(
            f"validation_months={validation_months} leaves no training data in {start}..{end}"
        )
    return {
        "train": (start, validation_start),
        "validation": (validation_start, midpoint),
        "test": (midpoint, end),
    }


def _read_pickled_zip(path: Path, member: str) -> pd.DataFrame | pd.Series:
    """Read a single pickled member out of one of the dataset's zip archives."""
    with zipfile.ZipFile(path) as archive:
        return pd.read_pickle(io.BytesIO(archive.read(member)))


def load_esa_channels(root: str | Path) -> pd.DataFrame:
    """
    Load ``channels.csv``, the channel metadata table.

    Arguments:
        root: Mission directory, e.g. ``.../ESA-Mission1``.
    Returns:
        pd.DataFrame: One row per channel, with a boolean ``is_target`` column
            added alongside the file's own ``YES``/``NO`` strings.
    """
    channels = pd.read_csv(Path(root) / "channels.csv")
    channels["is_target"] = channels["Target"].str.upper().eq("YES")
    return channels


def rescope_anomaly_types(
    labels: pd.DataFrame,
    *,
    cadence_s: int = MISSION1_CADENCE_S,
    point_max_samples: int = POINT_MAX_SAMPLES,
) -> pd.DataFrame:
    """
    Recompute the type attributes that depend on which channels are in scope.

    ``anomaly_types.csv`` describes each event across the *whole* mission, so an
    event listed as Multivariate may touch only one channel of a subset, and one
    listed as Subsequence may be a point anomaly within it. ESA-ADB says as much
    and ships ``infer_anomaly_types.py`` for the purpose; this is the same rule
    applied in-process.

    Arguments:
        labels: Segment rows, already filtered to the channels of interest.
        cadence_s: Dominant sampling period in seconds.
        point_max_samples: Longest event still counted as a point anomaly.
    Returns:
        pd.DataFrame: A copy with ``Dimensionality`` and ``Length`` recomputed
            for the channels present, and the originals preserved as
            ``Dimensionality_mission`` and ``Length_mission``.

    Notes:
        ``Locality`` is **not** recomputed and remains mission-scope. Deciding it
        requires the min/max of every nominal sample per channel, which means
        reading the full signal; inferring it from annotations alone is not
        possible. Treat that column as mission-wide wherever it is used.

        Length follows the benchmark's rule that a fragmented event counts as a
        point anomaly only if *every* one of its regions does.
    """
    out = labels.copy()
    out["Dimensionality_mission"] = out["Dimensionality"]
    out["Length_mission"] = out["Length"]

    affected = out.groupby("ID")["Channel"].transform("nunique")
    out["Dimensionality"] = np.where(affected > 1, "Multivariate", "Univariate")

    span_s = (out["EndTime"] - out["StartTime"]).dt.total_seconds()
    segment_is_point = span_s <= point_max_samples * cadence_s
    event_is_point = segment_is_point.groupby(out["ID"]).transform("all")
    out["Length"] = np.where(event_is_point, "Point", "Subsequence")
    return out


def load_esa_labels(
    root: str | Path,
    channels: Sequence[str] | None = None,
    *,
    drop_categories: Sequence[str] | None = NON_EVENT_CATEGORIES,
    rescope_types: bool = True,
    cadence_s: int = MISSION1_CADENCE_S,
) -> pd.DataFrame:
    """
    Load the annotations, joined to their event types.

    Arguments:
        root: Mission directory, e.g. ``.../ESA-Mission1``.
        channels: Keep only segments affecting these channels. None keeps all.
        drop_categories: Categories to exclude. Defaults to communication gaps,
            matching ESA-ADB's headline scoring, which reports every other
            category together. Pass None to keep everything, or
            ``("Communication Gap", "Rare Event")`` to score true anomalies only.
        rescope_types: Recompute ``Dimensionality`` and ``Length`` for the
            selected channels via :func:`rescope_anomaly_types`. Ignored when
            ``channels`` is None, where mission scope is already correct.
        cadence_s: Dominant sampling period, used only by the rescoping.
    Returns:
        pd.DataFrame: One row per *segment*, with columns ``ID``, ``Channel``,
            ``StartTime``, ``EndTime`` (tz-naive datetimes) and the event's
            ``Class``, ``Subclass``, ``Category``, ``Dimensionality``,
            ``Locality`` and ``Length``.

    Notes:
        Group by ``ID`` to recover events. A count of rows is a count of
        segments, which is a different and larger number: one event spans every
        channel it affects, and may be split into several disjoint regions per
        channel so that a burst of related disturbances still counts once.
    """
    root = Path(root)
    labels = pd.read_csv(root / "labels.csv", parse_dates=["StartTime", "EndTime"])
    # Timestamps are stored with a Z suffix; the channel index is tz-naive UTC,
    # so drop the tz rather than leave the two incomparable.
    for column in ("StartTime", "EndTime"):
        if isinstance(labels[column].dtype, pd.DatetimeTZDtype):
            labels[column] = labels[column].dt.tz_convert(None)

    labels = labels.merge(pd.read_csv(root / "anomaly_types.csv"), on="ID", how="left")
    if channels is not None:
        labels = labels[labels["Channel"].isin(list(channels))]
    if drop_categories:
        labels = labels[~labels["Category"].isin(list(drop_categories))]

    labels = labels.sort_values(["StartTime", "Channel"], ignore_index=True)
    if rescope_types and channels is not None:
        labels = rescope_anomaly_types(labels, cadence_s=cadence_s)
    return labels


def load_esa(
    root: str | Path,
    channels: Sequence[str] | None = None,
    *,
    split: str | None = None,
    span: tuple[pd.Timestamp, pd.Timestamp] = MISSION1_SPAN,
    validation_months: int = 3,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    resample_rule: str | None = None,
) -> TelemetryDataset:
    """
    Load telemetry channels into the canonical long form.

    Arguments:
        root: Mission directory, e.g. ``.../ESA-Mission1``.
        channels: Channel names to load, e.g. ``("channel_41", "channel_42")``.
            Defaults to :data:`MISSION1_LIGHTWEIGHT`.
        split: ``"train"``, ``"validation"`` or ``"test"``, resolved through
            :func:`esa_splits`. Use this rather than hand-written dates so
            results stay comparable with the published benchmark. Cannot be
            combined with ``start`` or ``end``.
        span: Mission span passed to :func:`esa_splits` when ``split`` is given.
        validation_months: Months of the training half held out for validation.
            The benchmark's default is 3; the paper also studies shorter
            training sets to find the earliest mission phase that supports a
            reliable detector.
        start: Keep samples at or after this timestamp. None for no lower bound.
        end: Keep samples strictly before this timestamp. None for no upper
            bound. Slicing happens per channel, before the frames are combined,
            so memory scales with the slice rather than the archive.
        resample_rule: Optional pandas offset alias (lowercase, e.g. ``"5min"``)
            to downsample onto a regular grid, taking the mean of each bin.
            None keeps the native cadence.
    Returns:
        TelemetryDataset: Long-form ``[timestamp, variable, value]``.

    Notes:
        Mission1 runs 14 years at a nominal 30 s cadence, so one channel is
        about 15.4 million samples. Six of them at full resolution is roughly a
        gigabyte once windowed, and a whole split is far more than that: prefer
        a slice within the split while iterating.

        Downsampling is not free, and on this dataset it is barely affordable
        at all: annotated events are short enough that a coarse rule erases
        them rather than blurring them. See :doc:`/user_guide/anomaly_scoring`
        for the measured cost at each rule, and prefer a slice at the native
        cadence over a resampled span.
    """
    root = Path(root)
    names = list(MISSION1_LIGHTWEIGHT if channels is None else channels)
    if not names:
        raise ValueError("channels must not be empty")

    if split is not None:
        if start is not None or end is not None:
            raise ValueError("Pass either split or start/end, not both.")
        splits = esa_splits(span, validation_months=validation_months)
        if split not in splits:
            raise ValueError(f"split must be one of {sorted(splits)}, got {split!r}")
        start, end = splits[split]

    start = None if start is None else pd.Timestamp(start)
    end = None if end is None else pd.Timestamp(end)

    frames = []
    for name in names:
        archive = root / "channels" / f"{name}.zip"
        if not archive.exists():
            raise FileNotFoundError(f"No archive for {name!r} at {archive}")

        series = _read_pickled_zip(archive, name)
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        # Slice before anything else: the archives are large and the caller is
        # usually after a fraction of the 14-year span.
        if start is not None or end is not None:
            series = series.loc[start:end]
            if end is not None and len(series) and series.index[-1] == end:
                series = series.iloc[:-1]  # `end` is exclusive
        if resample_rule is not None:
            series = series.resample(resample_rule).mean().dropna()

        frames.append(
            pd.DataFrame(
                {
                    _TS: series.index,
                    _VAR: name,
                    _VAL: series.to_numpy(dtype="float32"),
                }
            )
        )

    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values([_TS, _VAR], ignore_index=True)
    return TelemetryDataset(long)
