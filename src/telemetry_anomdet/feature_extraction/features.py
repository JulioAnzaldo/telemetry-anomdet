# telemetry_anomdet/features.py

"""
Feature extraction methods for telemetry analysis.

This module defines functions to compute relevant
features from preprocessed telemetry data.
"""

# Long form telemetry means one observation per row, wide means all variables get their own column, with timestamps as the index

import pandas as pd
import numpy as np

# Canonical column names
_TS, _VAR, _VAL = "timestamp", "variable", "value"

_REQURED_COLS = {_TS, _VAR, _VAL}

def pivot_wide(df: pd.DataFrame, *, variables = None) -> pd.DataFrame:
    """
    Convert long form telemetry data to wide format.
    
    Arguments:
        df (pd.DataFrame): Long-form data with ['timestamp','variable','value'].
        variables (list[str], optional): Specific variables to include; all by default.
    Returns:
        pd.DataFrame: Wide table (timestamp index, variables as columns).
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("pivot_table() expects a pandas DataFrame")
    
    missing = _REQURED_COLS - set(df.columns)

    if missing:
        raise ValueError(f"pivot_wide() missing required columns: {missing}")
    
    data = df

    # (Optional): filter to a subset of variables/sensors
    if variables is not None:
        data = data[data[_VAR].isin(variables)]

    if data.empty:
        return pd.DataFrame(index = pd.Index([], name = _TS))

    # Pivot from long to wide
    wide = (
        data.pivot_table(
            index = _TS,
            columns = _VAR,
            values = _VAL,
            aggfunc = "last",
        )
        .sort_index()
    )

    # Drop channels that are entirely NaN
    wide = wide.dropna(axis = 1, how = "all")

    # Deterministic column order
    if wide.shape[1] > 0:
        wide = wide.loc[:, sorted(wide.columns)]

    # Tidy up the labels
    wide.index.name = _TS
    wide.columns.name = None

    return wide

def windowify(wide_df: pd.DataFrame, *, window_size: int = 50, step: int = 10) -> np.ndarray:
    """
    Slice wide form telemetry data into overlapping windows.
    
    Arguments:
        wide_df (pd.DataFrame): Wide telemetry table.
        window_size (int): Number of samples per window.
        step (int): Step size between consecutive windows.
    Returns:
        np.ndarray: Array with shape (n_windows, window_size, n_features).
    """

    if not isinstance(wide_df, pd.DataFrame):
        raise TypeError("windoify() expects a pandas DataFrame")
    
    # Convert to numeric matrix: shape (n_samples, n_features)
    values = wide_df.to_numpy(dtype = float)
    n_samples, n_features = values.shape

    windows = []

    # Slide a fixed size window across time axis
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        windows.append(values[start:end, :])

    # If not enough data for a single window, return an empty array
    if not windows:
        return np.empty((0, window_size, n_features), dtype = float)

    # Stack list of (window_size, n_features) into (n_windows, window_size, n_features)
    return np.stack(windows, axis = 0)

def features_stat(X3d: np.ndarray) -> np.ndarray:
    """
    Generate a 2D feature table from windowed telemetry.

    Arguments:
        X3d (np.ndarray): Windowed telemetry tensor with shape
            (n_windows, window_size, n_features), typically produced by
            `make_feature_table()`.

    Returns:
        np.ndarray:
            Shape (n_windows, n_features * 4).
            For each window and each variable, this function computes
            simple statistical features (mean, standard deviation, min,
            and max) along the time axis and concatenates them into a
            flat feature vector ready for downstream modeling.
    """

    X3d = np.asarray(X3d)

    if X3d.ndim != 3:
        raise ValueError(
            f"features_stat() expects a 3D array "
            f"(n_windows, window_size, n_features), got {X3d.shape}"
        )

    n_windows, window_size, n_features = X3d.shape

    if n_windows == 0 or n_features == 0:
        # If no data return an empty 2D array with 0 rows
        return np.empty((0, 0), dtype = float)

    # Compute stats along the time axis (axis = 1)
    means = np.nanmean(X3d, axis = 1) # (n_windows, n_features)
    stds = np.nanstd(X3d, axis = 1) # (n_windows, n_features)
    mins = np.nanmin(X3d, axis = 1) # (n_windows, n_features)
    maxs = np.nanmax(X3d, axis = 1) # (n_windows, n_features)
    medians = np.nanmedian(X3d, axis=1) # (n_windows, n_features)

    # Linear slope per variable for each window
    times = np.arange(window_size)
    slopes = np.zeros((n_windows, n_features), dtype = float)

    for i in range(n_windows):
        for j in range(n_features):
            y = X3d[i, :, j]
            mask = np.isfinite(y)

            # If all NaN or fewer than 2 finite points, slope stays 0
            if mask.sum() < 2:
                slopes[i, j] = 0.0
            else:
                a, b = np.polyfit(times[mask], y[mask], 1)
                slopes[i, j] = a

    # Concatenate into a single feature vector per window: [mean | std | min | max]
    X_feat = np.concatenate([means, stds, mins, maxs, medians, slopes], axis = 1)

    # Replace any remaining NaN/inf with column means
    col_means = np.nanmean(X_feat, axis = 0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    bad = ~np.isfinite(X_feat)
    if bad.any():
        X_feat[bad] = col_means[np.where(bad)[1]]

    return X_feat

def make_feature_table(df: pd.DataFrame, *, variables = None, window_size = 50, step = 10) -> np.ndarray:
    """
    Generate a complete feature table from preprocessed telemetry.
    
    Arguments:
        df (pd.DataFrame): Preprocessed long-form telemetry data.
        variables (list[str], optional): Subset of sensors to include.
        window_size (int): Window length (default 50 samples).
        step (int): Step size between windows.
    Returns:
         np.ndarray:
            Shape (n_windows, window_size, n_features).
            Ready for downstream modeling or for passing into a
            future features_stat() to get a 2D feature table.
    """
    
    # Convert long form to wide form
    wide = pivot_wide(df, variables = variables)
    
    # If no usable data remains, stop early
    if wide.empty:
        return np.empty((0, window_size, 0))
    
    # Slice the wide data into fixed length windows
    X3d = windowify(wide, window_size = window_size, step = step)

    # Return the 3D telemetry tensor ready for feature extraction
    return X3d