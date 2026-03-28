"""Heuristic health classification and recommendation text for FRA diagnostics."""


def classify_from_correlation(corr: float) -> dict:
    """
    Map Pearson correlation between curves to status, severity, and a short recommendation.

    Parameters
    ----------
    corr
        Correlation coefficient in ``[-1, 1]`` (typically ``0`` to ``1`` for FRA).

    Returns
    -------
    dict
        Keys: ``status``, ``severity``, ``recommendation``.
    """
    if corr > 0.98:
        return {
            "status": "Healthy",
            "severity": "Low",
            "recommendation": (
                "Transformer operating within normal parameters. No action required."
            ),
        }
    if corr > 0.90:
        return {
            "status": "Warning",
            "severity": "Medium",
            "recommendation": (
                "Minor deviation detected. Schedule a DGA (Dissolved Gas Analysis) "
                "to confirm internal state."
            ),
        }
    return {
        "status": "Danger",
        "severity": "High",
        "recommendation": (
            "Significant frequency response shift! Immediate internal inspection "
            "of windings recommended."
        ),
    }


def get_recommendation(fault_type: str) -> str:
    """
    Return a detailed, actionable paragraph for a classified fault label.

    Parameters
    ----------
    fault_type
        Label from the ML model or a manual review bucket.

    Returns
    -------
    str
        Human-readable guidance for operators.
    """
    recommendations = {
        "Winding Deformation": (
            "Significant deviation detected in mid-frequency range (10kHz–100kHz) "
            "suggesting axial winding displacement. Schedule internal inspection within 30 days. "
            "Reduce loading to 80% capacity until inspection is completed."
        ),
        "Insulation Degradation": (
            "Minor capacitance changes in high-frequency region may indicate early-stage "
            "insulation aging. Perform dissolved gas analysis (DGA) to confirm. "
            "Continue monitoring with quarterly FRA tests."
        ),
        "Core Displacement": (
            "Slight low-frequency deviation within acceptable tolerance. May be due to "
            "measurement noise or minor core settling. No immediate action required. "
            "Include in next scheduled maintenance review."
        ),
        "Healthy": (
            "Signature matches reference baseline within 98% correlation. "
            "No mechanical or electrical anomalies detected. Resume standard annual monitoring."
        ),
    }

    return recommendations.get(
        fault_type,
        "Anomalous signature detected. Data requires manual review by a senior transformer engineer.",
    )


def get_severity_color(severity: str) -> str:
    """
    Map a coarse severity label to a hex color for dashboards.

    Parameters
    ----------
    severity
        One of ``High``, ``Medium``, ``Low``, or other labels.

    Returns
    -------
    str
        Hex color string (e.g. ``\"#ef4444\"``).
    """
    colors = {
        "High": "#ef4444",
        "Medium": "#f59e0b",
        "Low": "#22c55e",
    }
    return colors.get(severity, "#94a3b8")
