# telemetry_anomdet/features.py

"""
Feature extraction methods for telemetry analysis.

This module defines functions to compute relevant
features from preprocessed telemetry data.
"""

# Long form telemetry means one observation per row, wide means all variables get their own column, with timestamps as the index

def pivot_wide(df, *, variables = None):
    """
    Convert long form telemetry data to wide format.
    
    Arguments:
        df (pd.DataFrame): Long-form data with ['timestamp','variable','value'].
        variables (list[str], optional): Specific variables to include; all by default.
    Returns:
        pd.DataFrame: Wide table (timestamp index, variables as columns).
    """
    pass

def windowify(wide_df, *, window_size = 50, step = 10):
    """
    Slice wide form telemetry data into overlapping windows.
    
    Arguments:
        wide_df (pd.DataFrame): Wide telemetry table.
        window_size (int): Number of samples per window.
        step (int): Step size between consecutive windows.
    Returns:
        np.ndarray: Array with shape (n_windows, window_size, n_features).
    """
    pass

def features_stat(X3d):
    """
    Extract simple statistical features from each window.
    
    Arguments:
        X3d (np.ndarray): 3D telemetry data (n_windows × window_size × n_features).
    Returns:
        np.ndarray: 2D array of flattened feature vectors (e.g., mean, std, min, max per channel).
    """
    pass

def make_feature_table(df, *, variables = None, window_size = 50, step = 10):
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
    pass