import pytest

from telemetry_anomdet.ingest.csv_loader import (
    DEFAULT_ALIASES,
    coerce_long,
    is_long_form,
    load_from_csv,
    pick_time_column,
)

CANONICAL = ["timestamp", "variable", "value"]


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return str(path)


def _frame(dataset):
    return dataset.df if hasattr(dataset, "df") else dataset.data


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_long_form_passes_through(tmp_path):
    path = _write(
        tmp_path,
        "long.csv",
        """
        timestamp,variable,value
        2026-01-01T00:00:05Z,temp,1.7
        2026-01-01T00:00:00Z,temp,1.5
        """,
    )
    df = _frame(load_from_csv(path))
    assert list(df.columns) == CANONICAL
    assert len(df) == 2
    # coerce_long sorts by time, so the later row moves second.
    assert df["value"].tolist() == [1.5, 1.7]


def test_wide_form_is_melted(tmp_path):
    path = _write(
        tmp_path,
        "wide.csv",
        """
        time,temp,volt
        2026-01-01T00:00:00Z,1.5,3.3
        2026-01-01T00:00:05Z,1.7,3.4
        """,
    )
    df = _frame(load_from_csv(path))
    assert list(df.columns) == CANONICAL
    # Two timestamps by two measurement columns.
    assert len(df) == 4
    assert set(df["variable"]) == {"temp", "volt"}


def test_wide_form_honours_explicit_value_cols(tmp_path):
    path = _write(
        tmp_path,
        "wide.csv",
        """
        time,temp,volt,note
        2026-01-01T00:00:00Z,1.5,3.3,ok
        """,
    )
    df = _frame(load_from_csv(path, value_cols=["temp"]))
    assert set(df["variable"]) == {"temp"}


def test_aliases_are_resolved(tmp_path):
    path = _write(
        tmp_path,
        "alias.csv",
        """
        ts,sensor,reading
        2026-01-01T00:00:00Z,temp,1.5
        """,
    )
    df = _frame(load_from_csv(path))
    assert list(df.columns) == CANONICAL
    assert df["variable"].iloc[0] == "temp"


# ---------------------------------------------------------------------------
# Alias resolution must not depend on hash order
# ---------------------------------------------------------------------------


def test_alias_tables_are_ordered():
    """
    Regression guard. These were sets, and `next(a for a in aliases[role] ...)`
    picked between competing aliases by set iteration order, which CPython
    randomises per process. The same file could then resolve to a different
    column between runs, or rename onto an existing canonical column and raise
    "cannot assemble with duplicate keys".
    """
    for role, aliases in DEFAULT_ALIASES.items():
        assert isinstance(aliases, tuple), f"{role} aliases must be ordered"
        assert aliases[0] == role, f"{role} must list its canonical name first"


def test_canonical_column_wins_over_competing_alias(tmp_path):
    """A file carrying both 'time' and 'timestamp' resolves to 'timestamp'."""
    path = _write(
        tmp_path,
        "ambiguous_time.csv",
        """
        time,timestamp,variable,value
        2020-06-06T00:00:00Z,2026-01-01T00:00:00Z,temp,1.5
        """,
    )
    df = _frame(load_from_csv(path))
    assert list(df.columns) == CANONICAL
    assert df["timestamp"].iloc[0].year == 2026


def test_competing_variable_aliases_resolve_by_precedence(tmp_path):
    """'channel' outranks 'sensor', per the order in DEFAULT_ALIASES."""
    path = _write(
        tmp_path,
        "ambiguous_var.csv",
        """
        sensor,channel,timestamp,value
        from_sensor,from_channel,2026-01-01T00:00:00Z,1.5
        """,
    )
    df = _frame(load_from_csv(path))
    assert df["variable"].iloc[0] == "from_channel"


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def test_unparseable_rows_are_dropped(tmp_path):
    path = _write(
        tmp_path,
        "dirty.csv",
        """
        timestamp,variable,value
        2026-01-01T00:00:00Z,temp,1.5
        not-a-date,temp,1.6
        2026-01-01T00:00:10Z,temp,not-a-number
        """,
    )
    df = _frame(load_from_csv(path))
    assert len(df) == 1


def test_timestamps_are_utc(tmp_path):
    path = _write(
        tmp_path,
        "tz.csv",
        """
        timestamp,variable,value
        2026-01-01T00:00:00+05:00,temp,1.5
        """,
    )
    df = _frame(load_from_csv(path))
    assert str(df["timestamp"].dt.tz) == "UTC"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_missing_time_column_raises(tmp_path):
    path = _write(
        tmp_path,
        "no_time.csv",
        """
        alpha,beta
        1,2
        """,
    )
    with pytest.raises(KeyError, match="No time column found"):
        load_from_csv(path)


def test_unknown_time_col_raises(tmp_path):
    path = _write(
        tmp_path,
        "wide.csv",
        """
        time,temp
        2026-01-01T00:00:00Z,1.5
        """,
    )
    with pytest.raises(KeyError, match="time_col='nope' not found"):
        load_from_csv(path, time_col="nope")


def test_unknown_value_cols_raise(tmp_path):
    path = _write(
        tmp_path,
        "wide.csv",
        """
        time,temp
        2026-01-01T00:00:00Z,1.5
        """,
    )
    with pytest.raises(KeyError, match="value_cols not found"):
        load_from_csv(path, value_cols=["nope"])


def test_wide_file_with_only_a_time_column_raises(tmp_path):
    path = _write(
        tmp_path,
        "only_time.csv",
        """
        time
        2026-01-01T00:00:00Z
        """,
    )
    with pytest.raises(ValueError, match="No measurement columns"):
        load_from_csv(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_is_long_form():
    assert is_long_form(["timestamp", "variable", "value"], DEFAULT_ALIASES)
    assert is_long_form(["TS", "Sensor", "Reading"], DEFAULT_ALIASES)
    assert not is_long_form(["time", "temp", "volt"], DEFAULT_ALIASES)


def test_pick_time_column_is_case_insensitive():
    assert (
        pick_time_column(["Timestamp", "v"], time_col=None, aliases=DEFAULT_ALIASES) == "Timestamp"
    )


def test_pick_time_column_prefers_explicit():
    assert pick_time_column(["time", "odd"], time_col="odd", aliases=DEFAULT_ALIASES) == "odd"


def test_coerce_long_returns_canonical_columns():
    import pandas as pd

    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z"],
            "variable": [5],
            "value": ["1.5"],
            "extra": ["dropped"],
        }
    )
    out = coerce_long(df)
    assert list(out.columns) == CANONICAL
    # variable is coerced to string, value to numeric.
    assert out["variable"].iloc[0] == "5"
    assert out["value"].iloc[0] == 1.5
