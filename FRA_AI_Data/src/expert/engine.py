"""Rule-based expert system for FRA fault hypotheses and severity."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.features.fra_signal_features import extract_fra_signal_features


def _safe_ratio(num: float, den: float) -> float:
    """Return num/den with a small floor on the denominator."""
    return float(num) / (float(den) + 1e-12)


def _severity_from_deviation(dev_db: float, rel_energy_shift: float) -> str:
    """
    Map continuous deviations to Low / Medium / High severity.

    Parameters
    ----------
    dev_db
        Representative magnitude deviation in dB (non-negative).
    rel_energy_shift
        Relative change in band energy (|ratio-1|).
    """
    score = dev_db + 20.0 * rel_energy_shift
    if score < 1.5:
        return "Low"
    if score < 4.0:
        return "Medium"
    return "High"


def evaluate_expert_rules(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    *,
    max_deviation_db: Optional[float] = None,
) -> dict[str, Any]:
    """
    Apply heuristic FRA rules: peak shift, band-energy shifts, and correlation.

    Parameters
    ----------
    reference, test
        Frames with ``Frequency`` and ``Magnitude`` columns.
    max_deviation_db
        Optional pre-computed max |ref-test| on a common grid (dB). If omitted, computed.

    Returns
    -------
    dict
        ``primary_hypothesis``, ``severity``, ``triggers`` (list of str),
        ``details`` (metrics used by the UI or PDF).
    """
    f_r = np.asarray(reference["Frequency"], dtype=np.float64).ravel()
    m_r = np.asarray(reference["Magnitude"], dtype=np.float64).ravel()
    f_t = np.asarray(test["Frequency"], dtype=np.float64).ravel()
    m_t = np.asarray(test["Magnitude"], dtype=np.float64).ravel()

    f_min = max(float(f_r.min()), float(f_t.min()))
    f_max = min(float(f_r.max()), float(f_t.max()))
    if f_max <= f_min:
        grid = np.array([f_min])
    else:
        grid = np.linspace(f_min, f_max, min(400, max(50, len(f_t))))

    feats_r = extract_fra_signal_features(f_r, m_r)
    feats_t = extract_fra_signal_features(f_t, m_t, reference_frequency=f_r, reference_magnitude=m_r)

    el_r, em_r, eh_r = (
        feats_r["energy_low_band"],
        feats_r["energy_mid_band"],
        feats_r["energy_high_band"],
    )
    el_t, em_t, eh_t = (
        feats_t["energy_low_band"],
        feats_t["energy_mid_band"],
        feats_t["energy_high_band"],
    )

    pf_r = np.asarray(feats_r["peak_frequencies"], dtype=np.float64).ravel()
    pf_t = np.asarray(feats_t["peak_frequencies"], dtype=np.float64).ravel()

    peak_shift_hz = 0.0
    if pf_r.size and pf_t.size:
        peak_shift_hz = abs(float(pf_t[0]) - float(pf_r[0]))

    r_low = _safe_ratio(el_t, el_r)
    r_mid = _safe_ratio(em_t, em_r)
    r_high = _safe_ratio(eh_t, eh_r)
    m_ref_i = np.interp(grid, f_r, m_r)
    m_test_i = np.interp(grid, f_t, m_t)
    diff = m_test_i - m_ref_i
    max_dev = float(np.max(np.abs(diff))) if max_deviation_db is None else float(max_deviation_db)

    corr = feats_t.get("correlation_reference")
    if corr is None or not np.isfinite(corr):
        if diff.size >= 3:
            c = np.corrcoef(m_ref_i, m_test_i)[0, 1]
            corr = float(c) if np.isfinite(c) else 0.0
        else:
            corr = 0.0
    else:
        corr = float(corr)

    triggers: list[str] = []
    primary = "Healthy"

    # Peak shift → winding deformation (mid-band structure)
    if pf_r.size and pf_t.size:
        norm = max(float(np.median(grid)), 1.0)
        if peak_shift_hz / norm > 0.02:
            triggers.append("Peak shift → possible winding deformation")
            primary = "Winding Deformation"

    # High-frequency attenuation → insulation
    rel_high = abs(r_high - 1.0)
    if r_high < 0.85 and rel_high > 0.05:
        triggers.append("High-frequency attenuation → possible insulation issue")
        if primary == "Healthy":
            primary = "Insulation Degradation"

    # Low-frequency deviation → core / magnetic circuit
    rel_low = abs(r_low - 1.0)
    if rel_low > 0.12 or max_dev > 2.0 and r_low < 0.9:
        triggers.append("Low-frequency energy deviation → possible core or clamping issue")
        if primary == "Healthy":
            primary = "Core Displacement"

    if corr < 0.92 and max_dev > 1.0:
        triggers.append("Broadband magnitude deviation vs reference")

    rel_mid = abs(r_mid - 1.0)
    severity = _severity_from_deviation(max_dev, max(rel_low, rel_high, rel_mid))

    if not triggers:
        triggers.append("No strong rule-based triggers; compare with ML and anomaly outputs")

    return {
        "primary_hypothesis": primary,
        "severity": severity,
        "triggers": triggers,
        "details": {
            "peak_shift_hz": peak_shift_hz,
            "energy_ratio_low": r_low,
            "energy_ratio_mid": r_mid,
            "energy_ratio_high": r_high,
            "max_deviation_db": max_dev,
            "correlation": corr,
        },
    }
