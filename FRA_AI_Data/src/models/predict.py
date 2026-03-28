"""Backward-compatible wrapper around :mod:`src.models.prediction`."""

import pandas as pd

from src.models.prediction import predict_from_fra_pair


def predict_fault(data1: pd.DataFrame, data2: pd.DataFrame) -> tuple:
    """
    Predict fault class and confidence from two FRA curves.

    Parameters
    ----------
    data1, data2
        Reference and test frames with ``Frequency`` and ``Magnitude``.

    Returns
    -------
    tuple
        ``(predicted_label, confidence_percent)`` where confidence is in ``[0, 100]``.
    """
    out = predict_from_fra_pair(data1, data2)
    pct = round(float(out["confidence"]) * 100.0, 2)
    return out["fault"], pct
