"""End-to-end FRA processing: parse → features → ML → anomaly → expert → plots."""

from __future__ import annotations

import os
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.expert.engine import evaluate_expert_rules
from src.expert.rules import classify_from_correlation, get_recommendation
from src.features.feature_extractor import extract_features, preprocess_all
from src.features.fra_signal_features import (
    feature_dict_for_ui,
    generate_feature_insights,
)
from src.models.isolation_anomaly import ensure_anomaly_model, predict_anomaly, score_to_anomaly_0_100
from src.models.prediction import predict_from_fra_pair
from src.parser import load_fra_data
from src.utils.plotting import (
    plot_comparison_fra,
    plot_difference_fra,
    plot_single_fra,
)


def _coerce_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Frequency/Magnitude columns exist."""
    if "Frequency" not in df.columns or "Magnitude" not in df.columns:
        raise ValueError("DataFrame must contain 'Frequency' and 'Magnitude' columns.")
    return df.copy()


def _max_deviation_db(ref: pd.DataFrame, test: pd.DataFrame) -> float:
    """Max |magnitude difference| on a shared linear frequency grid (dB)."""
    f_r = np.asarray(ref["Frequency"], dtype=np.float64).ravel()
    m_r = np.asarray(ref["Magnitude"], dtype=np.float64).ravel()
    f_t = np.asarray(test["Frequency"], dtype=np.float64).ravel()
    m_t = np.asarray(test["Magnitude"], dtype=np.float64).ravel()
    f_min = max(float(f_r.min()), float(f_t.min()))
    f_max = min(float(f_r.max()), float(f_t.max()))
    if f_max <= f_min:
        return 0.0
    grid = np.linspace(f_min, f_max, min(400, max(50, len(f_t))))
    m_ref_i = np.interp(grid, f_r, m_r)
    m_test_i = np.interp(grid, f_t, m_t)
    return float(np.max(np.abs(m_test_i - m_ref_i)))


def _merge_severity(a: str, b: str) -> str:
    order = {"Low": 0, "Medium": 1, "High": 2}
    return a if order.get(a, 1) >= order.get(b, 1) else b


def _unified_diagnosis(
    *,
    ml_fault: str,
    ml_confidence_pct: float,
    expert: dict[str, Any],
    corr_rules: dict[str, Any],
    anomaly: dict[str, Any],
    anomaly_score_0_100: float,
    max_dev_db: float,
) -> dict[str, Any]:
    """Combine ML, expert engine, correlation rules, and anomaly into one UI object."""
    expert_primary = str(expert.get("primary_hypothesis", "Healthy"))
    # Prefer ML label when confidence is high; otherwise let expert primary hypothesis stand.
    fault = ml_fault
    if ml_confidence_pct < 55.0 and expert_primary != "Healthy":
        fault = expert_primary

    severity = _merge_severity(str(corr_rules.get("severity", "Medium")), str(expert.get("severity", "Medium")))
    if anomaly.get("is_anomaly") and severity == "Low":
        severity = "Medium"

    # Confidence: blend ML max-probability with correlation-based trust
    corr = float(expert.get("details", {}).get("correlation", 0.0))
    blended = 0.6 * ml_confidence_pct + 0.4 * float(np.clip(corr * 100.0, 0.0, 100.0))
    if max_dev_db > 6.0:
        blended = max(blended * 0.85, 40.0)

    rec_ml = get_recommendation(fault)
    rec_corr = str(corr_rules.get("recommendation", ""))
    recommendation = rec_ml if len(rec_ml) > len(rec_corr) else rec_corr
    if expert.get("triggers"):
        recommendation = f"{recommendation} Expert notes: " + "; ".join(expert["triggers"][:3])

    explanation_parts = [
        f"ML classifier: {ml_fault} ({ml_confidence_pct:.0f}% estimated class confidence).",
        f"Statistical correlation vs reference: {corr:.3f}.",
        f"IsolationForest anomaly score (0–100, higher=worse): {anomaly_score_0_100:.1f}.",
    ]
    if expert.get("triggers"):
        explanation_parts.append("Rules: " + " | ".join(expert["triggers"][:4]))

    return {
        "fault": fault,
        "confidence": float(round(blended, 1)),
        "severity": severity,
        "anomaly_score": float(round(anomaly_score_0_100, 3)),
        "recommendation": recommendation.strip(),
        "explanation": " ".join(explanation_parts),
    }


def process_fra_dataframes(
    ref_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    run_id: str,
    plot_root: str,
) -> dict[str, Any]:
    """
    Run ML, anomaly detection, expert rules, and plot generation on preprocessed frames.

    Parameters
    ----------
    ref_df, test_df
        Preprocessed data with ``Frequency`` and ``Magnitude``.
    run_id
        Unique suffix for plot file names.
    plot_root
        Directory where PNG assets are written.

    Returns
    -------
    dict
        Analysis payload (no ``file_path`` / parse metadata); raises on failure.
    """
    ref_df = preprocess_all(_coerce_dataframe(ref_df))
    test_df = preprocess_all(_coerce_dataframe(test_df))

    max_dev = _max_deviation_db(ref_df, test_df)

    ml_out = predict_from_fra_pair(ref_df, test_df)
    ml_conf_pct = round(float(ml_out["confidence"]) * 100.0, 2)

    feat_vec = extract_features(ref_df, test_df)
    if_model = ensure_anomaly_model()
    an = predict_anomaly(feat_vec, if_model)
    an_score = score_to_anomaly_0_100(float(an["score"]))

    f_t = np.asarray(test_df["Frequency"], dtype=np.float64)
    m_t = np.asarray(test_df["Magnitude"], dtype=np.float64)
    f_r = np.asarray(ref_df["Frequency"], dtype=np.float64)
    m_r = np.asarray(ref_df["Magnitude"], dtype=np.float64)

    test_features = feature_dict_for_ui(f_t, m_t, reference_frequency=f_r, reference_magnitude=m_r)
    ref_features = feature_dict_for_ui(f_r, m_r)
    insights = generate_feature_insights(test_features, reference_summary=ref_features)

    expert = evaluate_expert_rules(ref_df, test_df, max_deviation_db=max_dev)

    f_min = max(float(f_r.min()), float(f_t.min()))
    f_max = min(float(f_r.max()), float(f_t.max()))
    if f_max <= f_min:
        common = np.array([f_min])
    else:
        common = np.linspace(f_min, f_max, min(500, max(50, len(f_t))))
    m_ref_i = np.interp(common, f_r, m_r)
    m_test_i = np.interp(common, f_t, m_t)
    corr_val = float(np.corrcoef(m_ref_i, m_test_i)[0, 1]) if common.size >= 2 else 0.0
    if not np.isfinite(corr_val):
        corr_val = 0.0

    corr_rules = classify_from_correlation(corr_val)

    diagnosis = _unified_diagnosis(
        ml_fault=str(ml_out["fault"]),
        ml_confidence_pct=float(ml_conf_pct),
        expert=expert,
        corr_rules=corr_rules,
        anomaly=an,
        anomaly_score_0_100=an_score,
        max_dev_db=max_dev,
    )

    os.makedirs(plot_root, exist_ok=True)
    p_single = os.path.join(plot_root, f"fra_single_{run_id}.png")
    p_cmp = os.path.join(plot_root, f"fra_compare_{run_id}.png")
    p_diff = os.path.join(plot_root, f"fra_diff_{run_id}.png")

    plot_single_fra(test_df, p_single, title="FRA — Test measurement")
    plot_comparison_fra(ref_df, test_df, p_cmp)
    plot_difference_fra(ref_df, test_df, p_diff, deviation_threshold_db=3.0)

    return {
        "features": test_features,
        "insights": insights,
        "reference_features": ref_features,
        "ml": {
            "fault": str(ml_out["fault"]),
            "confidence": ml_conf_pct,
        },
        "anomaly": {
            "is_anomaly": bool(an["is_anomaly"]),
            "raw_score": float(an["score"]),
            "anomaly_score": an_score,
        },
        "expert": expert,
        "correlation": corr_val,
        "max_deviation_db": max_dev,
        "diagnosis": diagnosis,
        "plots": {
            "single": p_single,
            "comparison": p_cmp,
            "difference": p_diff,
            "single_url": "/static/plots/" + os.path.basename(p_single),
            "comparison_url": "/static/plots/" + os.path.basename(p_cmp),
            "difference_url": "/static/plots/" + os.path.basename(p_diff),
        },
    }


def process_fra(
    file_path: str,
    *,
    reference_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """
    Full diagnostic pipeline for one test FRA file.

    Steps: auto-parse test file, preprocess, extract features, classify (RF), run
    IsolationForest on paired features, evaluate expert rules, render plots.

    Parameters
    ----------
    file_path
        Path to uploaded test FRA (CSV/Excel).
    reference_path
        Optional baseline file. When omitted, uses ``data/raw/fra_healthy.csv`` if present,
        otherwise uses the test file as a degraded self-reference (limited diagnostics).
    output_dir
        Directory for PNG plots. Defaults to ``app/static/plots`` under the package root.

    Returns
    -------
    dict
        Structured result including ``features``, ``insights``, ``ml``, ``anomaly``,
        ``diagnosis`` (unified), ``plots``, ``expert``, and error metadata if any step fails.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    plot_root = output_dir or os.path.join(base_dir, "app", "static", "plots")
    os.makedirs(plot_root, exist_ok=True)

    default_ref = os.path.join(base_dir, "data", "raw", "fra_healthy.csv")
    ref_path = reference_path or (default_ref if os.path.isfile(default_ref) else file_path)

    result: dict[str, Any] = {
        "ok": False,
        "run_id": run_id,
        "file_path": os.path.abspath(file_path),
        "reference_path": os.path.abspath(ref_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }

    try:
        test_raw = load_fra_data(file_path)
        ref_raw = load_fra_data(ref_path)
        if test_raw is None:
            raise RuntimeError("Failed to parse test FRA file.")
        if ref_raw is None:
            raise RuntimeError("Failed to parse reference FRA file.")

        inner = process_fra_dataframes(ref_raw, test_raw, run_id=run_id, plot_root=plot_root)
        result.update({"ok": True, **inner})
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


__all__ = ["process_fra", "process_fra_dataframes"]
