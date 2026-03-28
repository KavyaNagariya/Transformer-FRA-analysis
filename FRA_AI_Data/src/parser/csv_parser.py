"""Load FRA measurements from CSV files with robust encoding and delimiter handling."""

from __future__ import annotations

import pandas as pd


def read_fra_csv(path: str) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame, trying UTF-8 first and falling back to latin-1
    with automatic delimiter detection when needed.

    Parameters
    ----------
    path
        Filesystem path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw tabular data as loaded from the file.
    """
    try:
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, encoding="latin1", sep=None, engine="python")
