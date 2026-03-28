"""Preprocessing and ML feature extraction for paired FRA curves."""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid entries and sort by increasing frequency.

    Parameters
    ----------
    data
        Frame with ``Frequency`` and ``Magnitude`` columns.

    Returns
    -------
    pd.DataFrame
        Cleaned, frequency-sorted data.
    """
    data = data.dropna()
    data = data.sort_values(by="Frequency").reset_index(drop=True)
    return data


def smooth_signal(data: pd.DataFrame, window: int = 11, polyorder: int = 3) -> pd.DataFrame:
    """
    Apply a Savitzky–Golay filter to magnitude when enough samples exist.

    Parameters
    ----------
    data
        Frame with a ``Magnitude`` column.
    window
        Filter window length (odd positive integer).
    polyorder
        Polynomial order for the filter.

    Returns
    -------
    pd.DataFrame
        Frame with smoothed ``Magnitude`` when applicable.
    """
    try:
        if len(data) > window:
            data = data.copy()
            data["Magnitude"] = savgol_filter(data["Magnitude"], window, polyorder)
        else:
            print(
                f"⚠️ Signal too short ({len(data)} pts) for window {window}. Skipping smoothing."
            )
    except Exception as e:
        print(f"Smoothing skipped: {e}")
    return data


def normalize_for_ai(data: pd.DataFrame) -> pd.DataFrame:
    """
    Min–max scale magnitude to ``[0, 1]`` as ``Magnitude_Scaled`` for ML inputs.

    Parameters
    ----------
    data
        Frame with ``Magnitude``.

    Returns
    -------
    pd.DataFrame
        Frame including ``Magnitude_Scaled``.
    """
    data = data.copy()
    mag = data["Magnitude"]
    data["Magnitude_Scaled"] = (mag - mag.min()) / (mag.max() - mag.min())
    return data


def preprocess_all(data: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline: clean, smooth, then normalize for the model.

    Parameters
    ----------
    data
        Raw FRA frame with ``Frequency`` and ``Magnitude``.

    Returns
    -------
    pd.DataFrame
        Processed frame ready for analysis and feature extraction.
    """
    data = clean_data(data)
    data = smooth_signal(data)
    data = normalize_for_ai(data)
    return data


def extract_features(data1: pd.DataFrame, data2: pd.DataFrame) -> list:
    """
    Interpolate two curves to 200 common frequency points and compute four statistics.

    Features are: mean absolute difference, std of difference, max difference, and correlation.

    Parameters
    ----------
    data1, data2
        Frames with ``Frequency`` and ``Magnitude`` (aligned scales preferred).

    Returns
    -------
    list
        ``[mean_diff, std_diff, max_diff, correlation]`` or a safe default on error.
    """
    try:
        f1 = np.array(data1["Frequency"])
        m1 = np.array(data1["Magnitude"])
        f2 = np.array(data2["Frequency"])
        m2 = np.array(data2["Magnitude"])

        f_min = max(min(f1), min(f2))
        f_max = min(max(f1), max(f2))

        if f_max <= f_min:
            return [0.0, 0.0, 0.0, 0.0]

        common_freq = np.linspace(f_min, f_max, 200)
        m1_interp = np.interp(common_freq, f1, m1)
        m2_interp = np.interp(common_freq, f2, m2)

        diff = np.abs(m1_interp - m2_interp)

        std1 = np.std(m1_interp)
        std2 = np.std(m2_interp)

        if std1 == 0 or std2 == 0:
            correlation = 0.0
        else:
            corr_matrix = np.corrcoef(m1_interp, m2_interp)
            correlation = corr_matrix[0, 1]
            if np.isnan(correlation):
                correlation = 0.0

        return [
            float(np.mean(diff)),
            float(np.std(diff)),
            float(np.max(diff)),
            float(correlation),
        ]

    except Exception as e:
        print(f"❌ Feature Extraction Error: {e}")
        return [10.0, 5.0, 20.0, 0.0]
