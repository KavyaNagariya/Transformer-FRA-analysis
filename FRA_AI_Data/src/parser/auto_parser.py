"""Auto-detect file format and normalize FRA columns to Frequency / Magnitude."""

from typing import Optional

import pandas as pd

from src.parser.universal_parser import FRAParseError, detect_fra_columns, parse_fra_file

# Backward-compatible name for column detection (delegates to universal parser).
detect_columns = detect_fra_columns


def load_fra_data(path: str) -> Optional[pd.DataFrame]:
    """
    Load FRA data from CSV or Excel, keep frequency/magnitude columns, and coerce numeric values.

    Uses :func:`parse_fra_file` internally and returns a two-column DataFrame.

    Parameters
    ----------
    path
        Path to a ``.csv`` or Excel file.

    Returns
    -------
    pd.DataFrame | None
        Two-column frame with ``Frequency`` and ``Magnitude``, or ``None`` on failure.
    """
    try:
        out = parse_fra_file(path)
        data = pd.DataFrame(
            {
                "Frequency": out["frequency"],
                "Magnitude": out["magnitude"],
            }
        )
        print(f"[OK] Data loaded successfully: {len(data)} points.")
        return data
    except FRAParseError as e:
        print(f"[ERROR] Data Loader Error ({path}): {e}")
        return None
