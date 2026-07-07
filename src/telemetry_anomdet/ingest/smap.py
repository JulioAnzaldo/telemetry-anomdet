# src/telemetry_anomdet/ingest/smap.py

"""
Loader for the NASA SMAP / MSL telemetry benchmark (telemanom format).

The benchmark ships one ``.npy`` array per channel under ``train/`` and
``test/`` directories, each of shape ``(timesteps, features)`` where column 0
is the telemetry value and the remaining columns are one-hot command context.
Anomaly labels live in ``labeled_anomalies.csv`` and are used for evaluation
only, never for training.

SMAP arrays carry no real timestamps, so this loader synthesizes a uniform
time index (configurable cadence). Output is the canonical long form
``[timestamp, variable, value]`` wrapped in a :class:`TelemetryDataset`.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from telemetry_anomdet.ingest.dataset import TelemetryDataset

# Canonical columns
_TS, _VAR, _VAL = "timestamp", "variable", "value"


def load_smap_labels(labels_csv: str | Path, *, spacecraft: str | None = "SMAP") -> pd.DataFrame:
    """
    Load and parse ``labeled_anomalies.csv``.

    Arguments:
        labels_csv: Path to the telemanom ``labeled_anomalies.csv``.
        spacecraft: Keep only rows for this spacecraft ('SMAP' or 'MSL').
            None keeps all rows.
    Returns:
        pd.DataFrame: The label rows with two added columns:
            'sequences' (list of ``(start, end)`` index tuples) and
            'anomaly_span' (total number of anomalous timesteps).
    """

    labels = pd.read_csv(labels_csv)
    if spacecraft is not None:
        labels = labels[labels["spacecraft"] == spacecraft].reset_index(drop=True)
        if len(labels) == 0:
            raise ValueError(f"No rows for spacecraft {spacecraft!r} in {labels_csv}")

    labels = labels.copy()
    labels["sequences"] = labels["anomaly_sequences"].apply(_parse_sequences)
    labels["anomaly_span"] = labels["sequences"].apply(
        lambda seqs: sum(end - start + 1 for start, end in seqs)
    )
    return labels


def anomaly_point_mask(sequences: Sequence[tuple[int, int]], n_timesteps: int) -> np.ndarray:
    """
    Build a point-level boolean mask from anomaly index ranges.

    Arguments:
        sequences: Iterable of inclusive ``(start, end)`` index ranges.
        n_timesteps: Length of the mask.
    Returns:
        np.ndarray: Boolean array of length ``n_timesteps``, True inside any range.
    """

    mask = np.zeros(int(n_timesteps), dtype=bool)
    for start, end in sequences:
        lo = max(0, int(start))
        hi = min(n_timesteps - 1, int(end))
        if hi >= lo:
            mask[lo : hi + 1] = True
    return mask


def load_smap_channel(
    npy_path: str | Path,
    chan_id: str | None = None,
    *,
    dims: str | Sequence[int] = "nonzero",
    cadence: str = "1s",
    start: str = "2000-01-01",
    tz: str = "UTC",
) -> TelemetryDataset:
    """
    Load a single SMAP channel ``.npy`` into a long-form TelemetryDataset.

    Arguments:
        npy_path: Path to the channel array of shape ``(timesteps, features)``.
        chan_id: Channel name used to prefix variables. Defaults to the file stem.
        dims: Which feature columns to keep. 'nonzero' drops all-zero columns
            (the default, matching the command one-hots that are inactive for a
            channel), 'all' keeps every column, 'telemetry' keeps only column 0,
            or pass an explicit list of column indices.
        cadence: Synthetic sampling interval (pandas offset alias, e.g. '1s').
        start: Synthetic start timestamp for the first sample.
        tz: Timezone for the synthesized index.
    Returns:
        TelemetryDataset: Long form with variables named ``f"{chan_id}_dim{j}"``.
    """

    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"SMAP channel file not found: {npy_path}")
    if chan_id is None:
        chan_id = npy_path.stem

    arr = np.load(npy_path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_timesteps = arr.shape[0]

    used_dims = _select_dims(arr, dims)
    timestamps = pd.date_range(start, periods=n_timesteps, freq=cadence, tz=tz)

    frames = [
        pd.DataFrame(
            {
                _TS: timestamps,
                _VAR: f"{chan_id}_dim{j}",
                _VAL: arr[:, j].astype(float),
            }
        )
        for j in used_dims
    ]
    long = pd.concat(frames, ignore_index=True)
    return TelemetryDataset(_coerce_long(long))


def load_smap(
    data_dir: str | Path,
    channels: Sequence[str],
    *,
    split: str = "test",
    dims: str | Sequence[int] = "nonzero",
    cadence: str = "1s",
    start: str = "2000-01-01",
    tz: str = "UTC",
) -> TelemetryDataset:
    """
    Load several SMAP channels into one combined long-form TelemetryDataset.

    Arguments:
        data_dir: Directory containing the ``train/`` and ``test/`` subfolders.
        channels: Channel ids to load (e.g. ['A-1', 'D-2']).
        split: 'train' or 'test'.
        dims, cadence, start, tz: Passed through to :func:`load_smap_channel`.
    Returns:
        TelemetryDataset: Combined long form; each channel keeps its own
            ``f"{chan_id}_dim{j}"`` variables so they never collide.
    """

    if not channels:
        raise ValueError("channels must be a non-empty sequence of channel ids")

    split_dir = Path(data_dir) / split
    frames = [
        load_smap_channel(
            split_dir / f"{chan_id}.npy",
            chan_id,
            dims=dims,
            cadence=cadence,
            start=start,
            tz=tz,
        ).to_pandas()
        for chan_id in channels
    ]
    long = pd.concat(frames, ignore_index=True)
    return TelemetryDataset(_coerce_long(long))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _parse_sequences(seq_str: str) -> list[tuple[int, int]]:
    """Parse an ``anomaly_sequences`` cell into a list of (start, end) tuples."""
    parsed = ast.literal_eval(seq_str) if isinstance(seq_str, str) else seq_str
    return [(int(a), int(b)) for a, b in parsed]


def _select_dims(arr: np.ndarray, dims: str | Sequence[int]) -> list[int]:
    """Resolve the ``dims`` argument to a concrete list of column indices."""
    n_dims = arr.shape[1]
    if isinstance(dims, str):
        if dims == "all":
            return list(range(n_dims))
        if dims == "telemetry":
            return [0]
        if dims == "nonzero":
            used = [j for j in range(n_dims) if np.any(arr[:, j] != 0.0)]
            return used or [0]  # fall back to the telemetry column if all zero
        raise ValueError(f"Unknown dims option: {dims!r}")
    used = [int(j) for j in dims]
    if not used:
        raise ValueError("dims list must not be empty")
    if any(j < 0 or j >= n_dims for j in used):
        raise ValueError(f"dims {used} out of range for array with {n_dims} columns")
    return used


def _coerce_long(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, type, and order a long-form frame (mirrors csv_loader.coerce_long)."""
    df[_TS] = pd.to_datetime(df[_TS], utc=True)
    df[_VAL] = pd.to_numeric(df[_VAL], errors="coerce")
    df[_VAR] = df[_VAR].astype(str)
    df = df.dropna(subset=[_TS, _VAR, _VAL]).sort_values([_TS, _VAR]).reset_index(drop=True)
    return df[[_TS, _VAR, _VAL]]
