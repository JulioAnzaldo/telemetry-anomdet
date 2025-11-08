# src/telemetry_anomdet/ingest/csv_loader.py

from __future__ import annotations
from telemetry_anomdet.ingest.dataset import TelemetryDataset
from typing import Optional, Sequence, Mapping, Iterable
import pandas as pd

# Canonical columns
_TS, _VAR, _VAL = "timestamp", "variable", "value"

# Common aliases
DEFAULT_ALIASES: Mapping[str, Iterable[str]] = {
    _TS: {"timestamp", "time", "datetime", "date", "ts"},
    _VAR: {"name", "key", "channel", "sensor", "variable"},
    _VAL: {"value", "reading", "val", "y"},
}

def load_from_csv(path: str, *, time_col: Optional[str] = None, value_cols: Optional[Sequence[str]] = None) -> TelemetryDataset:
    """
        Load telemetry from a CSV file.

        Notes:
        CSV must contain at least colums: timestamp, variable, value. timestamp will be converted to pandas datetime.

        Arguments:
        path - Path to CSV file.
        time_col - Explicit time column, if ommitted, guessed from aliases
        value_cols - Explicit value columns

        Returns:
        TelemetryDataset
    """
    
    df = pd.read_csv(path)
    aliases_local = DEFAULT_ALIASES
    
    # If in long form, normalize column names to canonical
    if is_long_form(df.columns.tolist(), aliases_local):
        ren = {}
        lowmap = {c.lower(): c for c in df.columns}
        ren[lowmap[next(a for a in aliases_local[_TS] if a in lowmap)]] = _TS
        ren[lowmap[next(a for a in aliases_local[_VAR] if a in lowmap)]] = _VAR
        ren[lowmap[next(a for a in aliases_local[_VAL] if a in lowmap)]] = _VAL

        out = df.rename(columns=ren)
        out = coerce_long(out)

        return TelemetryDataset(out)
    
    # If not in long form, convert from Wide to melt
    tcol = pick_time_column(df.columns.tolist(), time_col=time_col, aliases=aliases_local)
    wide = df.copy()
    wide[tcol] = pd.to_datetime(wide[tcol], errors="coerce", utc=True)
    wide = wide.dropna(subset=[tcol])

    # Choose measurement columns
    if value_cols:
        missing = [c for c in value_cols if c not in wide.columns]
        if missing:
            raise KeyError(f"value_cols not found: {missing}. Columns: {list(wide.columns)}")
        meas = list(value_cols)
    else:
        meas = [c for c in wide.columns if c != tcol]
        if not meas:
            raise ValueError("No measurement columns found. Provide value_cols=[].")

    long = wide.melt(id_vars = tcol, value_vars = meas, var_name = _VAR, value_name = _VAL)
    long = long.rename(columns = {tcol: _TS})
    long = coerce_long(long)

    return TelemetryDataset(long)

# Helper for finding datasets with timestamps/variables/values - returns t/f
def is_long_form(cols: Sequence[str], aliases: Mapping[str, Iterable[str]]) -> bool:
    """
    Determine whether a CSV is already in "long" form by checking that all three semantic roles appear under some alias.
    """
    
    lc = set(c.lower() for c in cols)
    has_ts = any(a in lc for a in aliases[_TS])
    has_var = any(a in lc for a in aliases[_VAR])
    has_val = any(a in lc for a in aliases[_VAL])
    
    return has_ts and has_var and has_val

def coerce_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleanup for a long form DataFrame
    - sort by time, then variable
    - ensure variable is string
    - parse timestamp to utc
    """

    df[_TS] = pd.to_datetime(df[_TS], errors="coerce", utc=True)
    df[_VAL] = pd.to_numeric(df[_VAL], errors="coerce")
    df[_VAR] = df[_VAR].astype(str)
    df = df.dropna(subset=[_TS, _VAR, _VAL]).sort_values([_TS, _VAR]).reset_index(drop=True)

    return df[[_TS, _VAR, _VAL]]

def pick_time_column(cols: Sequence[str], *, time_col: Optional[str], aliases: Mapping[str, Iterable[str]]) -> str:
    """
    Choose which colun is the time axis for *wide* CSV.
    - if time_col is explicitly provided, verify it exists and return it
    - else, try and match from alias candidates (like: "timestamp", "time")
    - if nothing can be found, raise error listing options and actual columns
    """

    if time_col:
        if time_col in cols:
            return time_col
        raise KeyError(f"time_col='{time_col}' not found in columns: {list(cols)}")
    
    lowered = {c.lower().strip(): c for c in cols}
    for cand in aliases.get(_TS, ()):
        if cand in lowered:
            return lowered[cand]
        
    raise KeyError(
        "No time column found. Provide time_col= or include one of the aliases: "
        f"{aliases.get(_TS)}. Columns seen: {list(cols)}"
    )