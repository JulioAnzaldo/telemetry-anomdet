# telemetry_anomdet/preprocessing.py

"""
Preprocessing utilities for telemetry data.

This module handles data cleaning, normalization,
and other transformations before feature extraction.
"""

# Long form telemetry means one observation per row, wide means all variables get their own column, with timestamps as the index

import fnmatch

import pandas as pd

# Canonical column names used throughout the preprocessing pipeline
_TS, _VAR, _VAL = "timestamp", "variable", "value"


def clean(df: pd.DataFrame, *, physical_bounds=None) -> pd.DataFrame:
    """
    Remove non existant values, non numeric readings, and physically impossible sensor values.

    Arguments:
        df (pd.DataFrame): Long form telemetry data with columns ['timestamp', 'variable', 'value'].
        physical_bounds (dict, optional): Mapping of variable names or patterns to (min, max) valid ranges.
            Example: {'Battery_Voltage': (0, 20), 'Battery_Temp': (-40, 85)}.
    Returns:
        pd.DataFrame: Cleaned dataset.
    """

    # Work on a copy
    df = df.copy()

    # Drop rows with missing core fields
    df = df.dropna(subset=[_TS, _VAR, _VAL])

    # Apply physical bounds
    if physical_bounds:
        # Mask that marks all rows as valid
        mask = pd.Series(True, index=df.index)

        # Loop over each variable pattern and its allowed (min, max) range
        for pattern, bounds in physical_bounds.items():
            if bounds is None:
                # Skip None bounds
                continue

            # Bounds must be a 2 element tuple/list (min/max)
            if not (isinstance(bounds, (tuple, list)) and len(bounds) == 2):
                raise ValueError("Must be (min, max)")

            lower, upper = bounds

            # Select rows whose variable name matches the given pattern.
            # Bind `pattern` as a default arg so the closure captures this
            # iteration's value, not the loop variable (ruff B023).
            sel = df[_VAR].map(lambda v, pattern=pattern: fnmatch.fnmatch(v, pattern))

            # Mask out rows that fall below or above the valid range
            if lower is not None:
                mask &= ~(sel & (df[_VAL] < float(lower)))
            if upper is not None:
                mask &= ~(sel & (df[_VAL] > float(upper)))

            # Apply the mask incrementally
            df = df[mask]

    # Deterministic ordering (by timestamp and variable)
    df = df.sort_values([_TS, _VAR]).reset_index(drop=True)

    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate or retransmitted rows.

    Arguments:
        df (pd.DataFrame): Long form telemetry data with potential duplicates.
    Returns:
        pd.DataFrame: DataFrame with duplicates (timestamp, variable) removed.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("dedupe() expects a pandas DataFrame")

    df = df.copy()

    # Sort so duplicates are grouped together
    df = df.sort_values([_TS, _VAR, _VAL])

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Drop retransmits (same timestamps, keep last occurence)
    df = df.drop_duplicates(subset=[_TS, _VAR], keep="last")

    # Reset index for cleanliness before returning
    return df.reset_index(drop=True)


def integrity_check(
    df: pd.DataFrame, *, require_utc: bool = True, require_sorted: bool = True
) -> None:
    """
    Verify timestamp format, timezone, and column consistency.

    Arguments:
        df (pd.DataFrame): Long form telemetry data.
        require_utc (bool): If True, ensure timestamps are UTC.
        require_sorted (bool): If True, ensure timestamps are sorted ascending.
    Raises:
        ValueError: If schema or ordering fails validation.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("integrity_check() expects a pandas DataFrame")

    # Extract timestamp column
    ts = df[_TS]

    # Ensure datetime (datetime64), if not try to coerce
    if not pd.api.types.is_datetime64_any_dtype(ts):
        try:
            ts = pd.to_datetime(ts, errors="raise", utc=require_utc)
        except Exception as e:
            raise ValueError("'timestamp' must be datetime format") from e

    # UTC requirement
    if require_utc:
        if ts.dt.tz is None:
            raise ValueError("Timestamps must be timezone aware UTC")
        if str(ts.dt.tz) not in ("UTC", "UTC+00:00", "tzutc()"):
            raise ValueError("Timestamps must be in UTC.")

    # Sorted requirement (chronological order if required)
    if require_sorted and not ts.is_monotonic_increasing:
        raise ValueError("Timestamps must be ascending.")

    # 'value' must be numeric
    if not pd.api.types.is_numeric_dtype(df[_VAL]):
        raise ValueError("'value' column must be numeric.")

    return


def resample(df: pd.DataFrame, *, rule: str = "5s", agg: str = "mean") -> pd.DataFrame:
    """
    Resample irregularly spaced data to a uniform cadence.

    Arguments:
        df (pd.DataFrame): Long form telemetry data.
        rule (str): Resample frequency ('1s', '5s', '1min').
        agg (str): Aggregation method ('mean', 'median', etc.) when multiple values exist per interval.
    Returns:
        pd.DataFrame: Resampled dataset with regular time intervals.
    """

    df = df.copy()

    # Ensure proper timestamp dtype
    df[_TS] = pd.to_datetime(df[_TS])

    # Pivot to wide form
    wide = df.pivot_table(index=_TS, columns=_VAR, values=_VAL)

    # Resample
    if agg == "mean":
        wide = wide.resample(rule).mean()
    elif agg == "median":
        wide = wide.resample(rule).median()
    elif agg == "min":
        wide = wide.resample(rule).min()
    elif agg == "max":
        wide = wide.resample(rule).max()
    else:
        raise ValueError(f"Unsupported agg method: {agg}")

    # Fill gaps with forward-fill then backfill
    # This reduces NaNs in windowing
    wide = wide.ffill().bfill()

    # Melt back to long form
    long = (
        wide.reset_index()
        .melt(id_vars=[_TS], var_name=_VAR, value_name=_VAL)
        .dropna(subset=[_VAL])  # remove variables missing entirely
        .sort_values(_TS)
        .reset_index(drop=True)
    )

    return long


def interpolate_gaps(
    df: pd.DataFrame, *, method: str = "ffill", limit: int | None = 1
) -> pd.DataFrame:
    """
    Fill small missing gaps to ensure continuous time steps.

    Arguments:
        df (pd.DataFrame): Resampled telemetry data in long form.
        method (str): Interpolation strategy ('ffill', 'bfill', 'linear', etc.).
        limit (int, optional): Maximum consecutive non existant value steps to fill.
            None means no limit.
    Returns:
        pd.DataFrame: Gap filled dataset in long form.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("interpolate_gaps() expects a pandas DataFrame")

    df = df.copy()
    df[_TS] = pd.to_datetime(df[_TS])

    # Pivot to wide form so each variable is a contiguous time series
    wide = df.pivot_table(index=_TS, columns=_VAR, values=_VAL)

    # Fill gaps column by column, capped at `limit` consecutive steps
    if method == "ffill":
        wide = wide.ffill(limit=limit)
    elif method == "bfill":
        wide = wide.bfill(limit=limit)
    else:
        # Time-aware interpolation methods (e.g. 'linear', 'time', 'spline')
        try:
            wide = wide.interpolate(method=method, limit=limit, limit_direction="forward")
        except (ValueError, NotImplementedError) as e:
            raise ValueError(f"Unsupported interpolation method: {method}") from e

    # Melt back to long form, dropping gaps that remained unfilled
    long = (
        wide.reset_index()
        .melt(id_vars=[_TS], var_name=_VAR, value_name=_VAL)
        .dropna(subset=[_VAL])
        .sort_values([_TS, _VAR])
        .reset_index(drop=True)
    )

    return long


def normalize_fit(df: pd.DataFrame, *, method: str = "zscore") -> dict:
    """
    Compute normalization parameters for each variable.

    Arguments:
        df (pd.DataFrame): Cleaned telemetry data (usually training subset).
        method (str): Normalization method ('zscore' or 'minmax').
    Returns:
        dict: Mapping {variable: (center, scale)} where the pair is
            (mean, std) for 'zscore' or (min, range) for 'minmax'.
            A scale of 0 (constant variable) is stored as 1.0 so that
            applying the parameters never divides by zero.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("normalize_fit() expects a pandas DataFrame")
    if method not in ("zscore", "minmax"):
        raise ValueError(f"Unsupported normalization method: {method}")

    params: dict = {}

    # Compute per-variable statistics independently
    for variable, group in df.groupby(_VAR):
        values = group[_VAL].astype(float)

        if method == "zscore":
            center = float(values.mean())
            # Sample std (ddof = 1); guard against constant/single-sample series
            scale = float(values.std(ddof=1))
        else:  # minmax
            center = float(values.min())
            scale = float(values.max() - center)

        # Avoid divide-by-zero for constant variables when params are applied
        if not scale > 0:
            scale = 1.0

        params[variable] = (center, scale)

    return params


def pipeline(
    df: pd.DataFrame,
    *,
    physical_bounds: dict | None = None,
    resample_rule: str | None = "5s",
    resample_agg: str = "mean",
    interpolate_method: str = "ffill",
    gap_limit: int | None = 1,
) -> pd.DataFrame:
    """
    Execute minimal preprocessing pipeline for this dataset.

    Steps: clean -> dedupe -> integrity_check -> resample -> interpolate_gaps.

    Normalization is intentionally kept out of this pipeline. Following
    scikit-learn convention, fitting normalization stats is a separate step:
    call :func:`normalize_fit` on the returned (training) frame and reuse the
    resulting params at inference to prevent leakage.

    Arguments:
        df (pd.DataFrame): Raw telemetry dataset.
        physical_bounds (dict, optional): Min/max physical limits per variable.
        resample_rule (str): Resampling frequency (default '5s'). None skips resampling.
        resample_agg (str): Aggregation method (default 'mean').
        interpolate_method (str): Gap-fill strategy ('ffill', 'linear', etc.).
        gap_limit (int, optional): Max consecutive steps to interpolate.
    Returns:
        pd.DataFrame: Fully preprocessed dataset.
    """

    # Remove nulls, non physical readings, and sort data
    df = clean(df, physical_bounds=physical_bounds)

    # Remove dupes or retransmits
    df = dedupe(df)

    # Validate schema/ordering before time-based operations
    integrity_check(df, require_utc=False, require_sorted=False)

    # Resample to a uniform cadence, then fill remaining small gaps
    if resample_rule is not None:
        df = resample(df, rule=resample_rule, agg=resample_agg)
        df = interpolate_gaps(df, method=interpolate_method, limit=gap_limit)

    return df
