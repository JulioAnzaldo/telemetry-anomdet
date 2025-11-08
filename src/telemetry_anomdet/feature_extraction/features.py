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

def pivot_wide(df: pd.DataFrame, *, variables = None) -> pd.DataFrame:
    """
    Convert long form telemetry data to wide format.
    
    Arguments:
        df (pd.DataFrame): Long-form data with ['timestamp','variable','value'].
        variables (list[str], optional): Specific variables to include; all by default.
    Returns:
        pd.DataFrame: Wide table (timestamp index, variables as columns).
    """
    
    data = df.copy()

    # (Optional): filter to a subset of variables/sensors
    if variables is not None:
        data = data[data[_VAR].isin(variables)]

    # Pivot from long to wide
    wide = (
        data
        .pivot_table(
            index = _TS,
            columns = _VAR,
            values = _VAL,
            aggfunc = "last",
        )
        .sort_index
    )

    # Ensure consistent columnd order
    wide = wide.loc[:, sorted(wide.columns)]

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

def features_stat(X3d):
    """
    Extract simple statistical features from each window.
    
    Arguments:
        X3d (np.ndarray): 3D telemetry data (n_windows × window_size × n_features).
    Returns:
        np.ndarray: 2D array of flattened feature vectors (e.g., mean, std, min, max per channel).
    """
    pass

def make_feature_table(df: pd.DataFrame, *, variables = None, window_size = 50, step = 10):
    """
    Generate a complete feature table from preprocessed telemetry.
    
    Arguments:
        df (pd.DataFrame): Preprocessed long-form telemetry data.
        variables (list[str], optional): Subset of sensors to include.
        window_size (int): Window length (default 50 samples).
        step (int): Step size between windows.
    Returns:
        pd.DataFrame: Feature table ready for model training (n_windows × n_features).
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