"""Matplotlib helpers for FRA overlays, difference maps, and static plot assets."""

from __future__ import annotations

import base64
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "app", "static", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def _ensure_fm(df: pd.DataFrame) -> pd.DataFrame:
    if "Frequency" not in df.columns or "Magnitude" not in df.columns:
        raise ValueError("Expected columns Frequency and Magnitude.")
    return df


def plot_single_fra(
    data: pd.DataFrame,
    out_path: str,
    *,
    title: str = "Frequency Response Analysis",
) -> str:
    """
    Plot magnitude vs frequency (log-x) for a single FRA curve and save PNG.

    Parameters
    ----------
    data
        Frame with ``Frequency`` and ``Magnitude``.
    out_path
        Destination ``.png`` path.
    title
        Plot title.

    Returns
    -------
    str
        Path written.
    """
    d = _ensure_fm(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d["Frequency"], d["Magnitude"], color="#38bdf8", linewidth=1.8, label="Magnitude")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    return out_path


def plot_comparison_fra(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    out_path: str,
    *,
    ref_label: str = "Reference",
    test_label: str = "Test (uploaded)",
) -> str:
    """
    Overlay reference vs test FRA curves with legend and log frequency axis.

    Parameters
    ----------
    reference, test
        Frames with ``Frequency`` and ``Magnitude``.
    out_path
        Output PNG path.

    Returns
    -------
    str
        Path written.
    """
    r = _ensure_fm(reference)
    t = _ensure_fm(test)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(r["Frequency"], r["Magnitude"], color="#38bdf8", linewidth=1.5, linestyle="--", label=ref_label)
    ax.plot(t["Frequency"], t["Magnitude"], color="#f59e0b", linewidth=2.0, label=test_label)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("FRA comparison — reference vs test")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    return out_path


def plot_difference_fra(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    out_path: str,
    *,
    n_points: int = 500,
    deviation_threshold_db: float = 3.0,
) -> str:
    """
    Plot ``test - reference`` magnitude (dB) on a common frequency grid.

    Highlights regions where |difference| exceeds ``deviation_threshold_db``.

    Parameters
    ----------
    reference, test
        Frames with ``Frequency`` and ``Magnitude``.
    out_path
        Output PNG path.
    n_points
        Number of linearly spaced samples on the overlap band.
    deviation_threshold_db
        Threshold for shaded deviation bands.

    Returns
    -------
    str
        Path written.
    """
    r = _ensure_fm(reference)
    t = _ensure_fm(test)
    f_r = np.asarray(r["Frequency"], dtype=np.float64).ravel()
    m_r = np.asarray(r["Magnitude"], dtype=np.float64).ravel()
    f_t = np.asarray(t["Frequency"], dtype=np.float64).ravel()
    m_t = np.asarray(t["Magnitude"], dtype=np.float64).ravel()

    f_min = max(float(f_r.min()), float(f_t.min()))
    f_max = min(float(f_r.max()), float(f_t.max()))
    if f_max <= f_min:
        grid = np.array([f_min])
    else:
        grid = np.linspace(f_min, f_max, n_points)
    m_ref_i = np.interp(grid, f_r, m_r)
    m_test_i = np.interp(grid, f_t, m_t)
    diff = m_test_i - m_ref_i

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grid, diff, color="#a78bfa", linewidth=1.8, label="Test − Reference (dB)")
    ax.axhline(0.0, color="#64748b", linewidth=1.0)
    ax.axhline(deviation_threshold_db, color="#ef4444", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.axhline(-deviation_threshold_db, color="#ef4444", linestyle=":", linewidth=1.0, alpha=0.8)

    mask = np.abs(diff) > deviation_threshold_db
    if np.any(mask):
        ax.fill_between(grid, diff, 0, where=mask, color="#ef4444", alpha=0.15, label="|Δ| > threshold")

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Δ Magnitude (dB)")
    ax.set_title("FRA deviation map (test minus reference)")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    return out_path


def generate_comparison_plot(data1: pd.DataFrame, data2: pd.DataFrame) -> str:
    """
    Build an overlay plot of two FRA curves and return a base64 data URL for HTML embedding.

    Parameters
    ----------
    data1, data2
        Frames with ``Frequency`` and ``Magnitude`` (typically baseline vs test).

    Returns
    -------
    str
        ``data:image/png;base64,...`` string suitable for ``<img src="...">``.
    """
    plt.figure(figsize=(10, 5))
    plt.style.use("dark_background")

    plt.plot(
        data1["Frequency"],
        data1["Magnitude"],
        label="Baseline (Healthy)",
        color="#38bdf8",
        linewidth=1.5,
        linestyle="--",
    )
    plt.plot(
        data2["Frequency"],
        data2["Magnitude"],
        label="Test Measurement",
        color="#f59e0b",
        linewidth=2,
    )

    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Frequency Response Analysis - Comparison")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.1)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close()
    buf.seek(0)

    plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{plot_data}"


def save_fra_plot(data: pd.DataFrame, filename: str = "latest_plot.png") -> str:
    """
    Save a single-curve FRA plot under ``app/static/plots``.

    Parameters
    ----------
    data
        Frame with ``Frequency`` and ``Magnitude``.
    filename
        Output file name inside the static plots directory.

    Returns
    -------
    str
        Full path to the written PNG file.
    """
    path = os.path.join(PLOT_DIR, filename)
    return plot_single_fra(data, path, title="FRA — Magnitude")


__all__ = [
    "PLOT_DIR",
    "generate_comparison_plot",
    "plot_comparison_fra",
    "plot_difference_fra",
    "plot_single_fra",
    "save_fra_plot",
]
