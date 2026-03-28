"""Load FRA measurements from Excel workbooks."""

from __future__ import annotations

import pandas as pd


def read_fra_excel(path: str) -> pd.DataFrame:
    """
    Read the first worksheet from an Excel file into a DataFrame.

    Parameters
    ----------
    path
        Filesystem path to the ``.xlsx`` (or other pandas-supported Excel) file.

    Returns
    -------
    pd.DataFrame
        Raw tabular data as loaded from the workbook.
    """
    return pd.read_excel(path)
