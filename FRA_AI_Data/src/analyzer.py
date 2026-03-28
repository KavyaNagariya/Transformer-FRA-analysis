"""High-level FRA diagnostic pipeline combining statistics, ML, anomaly, and expert rules."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.pipeline import process_fra_dataframes


def advanced_analysis(
    healthy_df: pd.DataFrame,
    uploaded_df: pd.DataFrame,
    *,
    plot_root: str | None = None,
) -> dict:
    """
    Run statistical metrics, ML fault classification, anomaly scoring, and expert rules.

    Parameters
    ----------
    healthy_df
        Baseline (reference) FRA measurement.
    uploaded_df
        Test or uploaded FRA measurement.
    plot_root
        Optional directory for matplotlib PNG outputs.

    Returns
    -------
    dict
        Keys include unified ``diagnosis``, ``features``, ``insights``, ``plots`` URLs,
        chart series ``frequencies``, ``magnitude_healthy``, ``magnitude_uploaded``,
        and legacy fields ``status``, ``severity``, ``fault_type``, ``confidence``,
        ``recommendation``, ``correlation``, ``shift``.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    root = plot_root or os.path.join(base_dir, "app", "static", "plots")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    core = process_fra_dataframes(healthy_df, uploaded_df, run_id=run_id, plot_root=root)
    diagnosis = core["diagnosis"]
    expert = core["expert"]

    # Chart.js: aligned curves on overlap band
    h = healthy_df.copy()
    u = uploaded_df.copy()
    f_r = np.asarray(h["Frequency"], dtype=np.float64).ravel()
    m_r = np.asarray(h["Magnitude"], dtype=np.float64).ravel()
    f_t = np.asarray(u["Frequency"], dtype=np.float64).ravel()
    m_t = np.asarray(u["Magnitude"], dtype=np.float64).ravel()
    f_min = max(float(f_r.min()), float(f_t.min()))
    f_max = min(float(f_r.max()), float(f_t.max()))
    if f_max <= f_min:
        grid = np.array([f_min])
    else:
        grid = np.linspace(f_min, f_max, min(600, max(50, len(f_t))))
    mag_h = np.interp(grid, f_r, m_r).tolist()
    mag_u = np.interp(grid, f_t, m_t).tolist()
    freq = grid.tolist()

    corr = float(core["correlation"])
    if corr > 0.98:
        status = "Healthy"
    elif corr > 0.90:
        status = "Warning"
    else:
        status = "Danger"

    return {
        "status": status,
        "shift": core["max_deviation_db"],
        "correlation": corr,
        "severity": diagnosis["severity"],
        "fault_type": diagnosis["fault"],
        "confidence": diagnosis["confidence"],
        "frequencies": freq,
        "magnitude_healthy": mag_h,
        "magnitude_uploaded": mag_u,
        "recommendation": diagnosis["recommendation"],
        "explanation": diagnosis["explanation"],
        "features": core["features"],
        "insights": core["insights"],
        "reference_features": core["reference_features"],
        "plots": core["plots"],
        "ml": core["ml"],
        "anomaly": core["anomaly"],
        "expert": expert,
        "diagnosis": diagnosis,
    }
