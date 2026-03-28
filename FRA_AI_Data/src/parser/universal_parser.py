"""Universal FRA file parser: CSV / Excel, auto format detection, standardized numpy output."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import pandas as pd

from src.parser.csv_parser import read_fra_csv
from src.parser.excel_parser import read_fra_excel


class FRAParseError(Exception):
    """Raised when a file cannot be parsed as frequency–magnitude FRA data."""


def _freq_column_score(name: str) -> int:
    """
    Score how likely a column is the frequency axis (higher = better match).

    Avoids treating unrelated tokens (e.g. substrings inside other words) as strong signals.
    """
    s = str(name).lower().strip()
    if "ghz" in s:
        return 0
    if "frequency" in s:
        return 100
    if re.search(r"\bfreq\b", s):
        return 80
    if re.search(r"\bhz\b", s) or re.search(r"\(\s*hz\s*\)", s):
        return 60
    if s == "hz" or s.endswith(" hz"):
        return 50
    if "hz" in s:
        return 25
    return 0


def _mag_column_score(name: str) -> int:
    """Score how likely a column is the magnitude axis (higher = better match)."""
    s = str(name).lower().strip()
    if "magnitude" in s:
        return 100
    if re.search(r"\bmag\b", s):
        return 85
    if "amplitude" in s:
        return 70
    if "gain" in s:
        return 55
    if re.search(r"\bdb\b", s) or "(db)" in s or " db" in s:
        return 75
    if s == "db" or s.endswith(" db"):
        return 65
    if "db" in s:
        return 40
    return 0


def detect_fra_columns(columns: Any) -> tuple[str, str]:
    """
    Pick frequency and magnitude columns from arbitrary column labels.

    Uses scored heuristics for names such as Frequency, Freq, Hz, Magnitude, Mag, dB.

    Parameters
    ----------
    columns
        Iterable of column names (e.g. ``DataFrame.columns``).

    Returns
    -------
    tuple[str, str]
        ``(frequency_column_name, magnitude_column_name)``.

    Raises
    ------
    FRAParseError
        If fewer than two columns exist or distinct columns cannot be chosen.
    """
    cols = [str(c) for c in columns]
    if len(cols) < 2:
        raise FRAParseError(
            f"Need at least two columns for frequency and magnitude; found {len(cols)}."
        )

    freq_scores = [(c, _freq_column_score(c)) for c in cols]
    mag_scores = [(c, _mag_column_score(c)) for c in cols]

    freq_col = max(freq_scores, key=lambda x: x[1])[0]
    mag_col = max(mag_scores, key=lambda x: x[1])[0]

    if freq_col == mag_col:
        best_freq = max(freq_scores, key=lambda x: x[1])
        best_mag = max(mag_scores, key=lambda x: x[1])
        if best_freq[1] == 0 and best_mag[1] == 0:
            freq_col, mag_col = cols[0], cols[1]
        elif best_freq[1] >= best_mag[1]:
            candidates = [c for c in cols if c != freq_col]
            mag_col = max(candidates, key=lambda c: _mag_column_score(c)) if candidates else mag_col
        else:
            candidates = [c for c in cols if c != mag_col]
            freq_col = max(candidates, key=lambda c: _freq_column_score(c)) if candidates else freq_col

    if freq_col == mag_col:
        freq_col, mag_col = cols[0], cols[1]

    return freq_col, mag_col


def _read_raw_frame(path: str) -> pd.DataFrame:
    """Load a tabular file based on extension; raises FRAParseError on failure."""
    if not os.path.isfile(path):
        raise FRAParseError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".xlsx", ".xls", ".xlsm"):
            return read_fra_excel(path)
        if ext == ".csv" or ext == "":
            return read_fra_csv(path)
        raise FRAParseError(
            f"Unsupported file type {ext!r}. Use .csv or Excel (.xlsx, .xls, .xlsm)."
        )
    except FRAParseError:
        raise
    except Exception as e:
        raise FRAParseError(f"Could not read file {path!r}: {e}") from e


def parse_fra_file(path: str) -> dict[str, np.ndarray]:
    """
    Parse an FRA measurement file into standardized numpy arrays.

    Accepts CSV and Excel (.xlsx / .xls / .xlsm). Format is inferred from the extension.
    Frequency and magnitude columns are detected from flexible names (e.g. Freq, Hz, Mag, dB).

    Parameters
    ----------
    path
        Path to a CSV or Excel file.

    Returns
    -------
    dict
        ``{"frequency": np.ndarray, "magnitude": np.ndarray}`` (float64, same length),
        sorted by increasing frequency. Arrays are 1-D.

    Raises
    ------
    FRAParseError
        If the file is missing, unreadable, not tabular FRA data, or contains no valid points.
    """
    raw = _read_raw_frame(path)
    if raw is None or raw.empty:
        raise FRAParseError("File is empty or could not be loaded as a table.")

    raw = raw.dropna(axis=1, how="all")
    if raw.shape[1] < 2:
        raise FRAParseError(
            f"Need at least two data columns; found {raw.shape[1]} after removing empty columns."
        )

    freq_name, mag_name = detect_fra_columns(raw.columns)
    frame = raw[[freq_name, mag_name]].copy()

    frame.iloc[:, 0] = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    frame.iloc[:, 1] = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    frame = frame.dropna()
    frame = frame.sort_values(by=frame.columns[0]).reset_index(drop=True)

    if frame.empty:
        raise FRAParseError("No valid numeric frequency/magnitude rows after cleaning.")

    frequency = frame.iloc[:, 0].to_numpy(dtype=np.float64)
    magnitude = frame.iloc[:, 1].to_numpy(dtype=np.float64)

    return {"frequency": frequency, "magnitude": magnitude}


def parse_fra_file_safe(path: str) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """
    Same as :func:`parse_fra_file` but return ``(data, None)`` or ``(None, error_message)``.
    """
    try:
        return parse_fra_file(path), None
    except FRAParseError as e:
        return None, str(e)
