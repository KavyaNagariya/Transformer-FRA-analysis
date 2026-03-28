"""Pointwise statistical comparison between two FRA magnitude traces."""

import numpy as np
import pandas as pd


def calculate_metrics(data1: pd.DataFrame, data2: pd.DataFrame) -> tuple:
    """
    Compare aligned magnitude arrays: peak index shift, max absolute deviation, correlation.

    Parameters
    ----------
    data1, data2
        Frames with ``Magnitude`` columns; lengths are synchronized to the shorter series.

    Returns
    -------
    tuple
        ``(peak_index_shift, max_deviation_db, correlation)`` — zeros on failure.
    """
    try:
        m1 = np.array(data1["Magnitude"])
        m2 = np.array(data2["Magnitude"])

        min_len = min(len(m1), len(m2))
        m1_sync = m1[:min_len]
        m2_sync = m2[:min_len]

        peak1 = np.argmax(m1_sync)
        peak2 = np.argmax(m2_sync)
        shift = int(abs(peak1 - peak2))

        deviation_array = m1_sync - m2_sync
        max_dev = np.max(np.abs(deviation_array))

        corr = np.corrcoef(m1_sync, m2_sync)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        return shift, round(float(max_dev), 2), round(float(corr), 4)
    except Exception as e:
        print(f"Metrics Calculation Error: {e}")
        return 0, 0.0, 0.0
