"""Synthetic FRA curve generation for demos, tests, and training aids."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _logspace_hz(n: int = 400, f_min: float = 20.0, f_max: float = 2e6) -> np.ndarray:
    return np.logspace(np.log10(f_min), np.log10(f_max), n)


def _base_curve(f: np.ndarray) -> np.ndarray:
    """Smooth dB-like magnitude with a few resonance bumps."""
    x = np.log10(np.clip(f, 1e-6, None))
    mag = (
        -12.0
        - 8.0 * np.exp(-((x - 3.2) ** 2) / 0.08)
        - 6.0 * np.exp(-((x - 4.1) ** 2) / 0.06)
        - 4.0 * np.exp(-((x - 5.0) ** 2) / 0.05)
        + 0.4 * np.sin(x * 6.0)
    )
    return mag.astype(np.float64)


def generate_healthy_fra(n: int = 400) -> pd.DataFrame:
    """
    Create a synthetic healthy FRA trace.

    Returns
    -------
    pd.DataFrame
        Columns ``Frequency``, ``Magnitude`` (dB).
    """
    f = _logspace_hz(n)
    m = _base_curve(f) + np.random.default_rng(42).normal(0.0, 0.15, size=f.size)
    return pd.DataFrame({"Frequency": f, "Magnitude": m})


def generate_winding_deformation_fra(
    healthy: pd.DataFrame | None = None,
    *,
    peak_shift_factor: float = 1.08,
) -> pd.DataFrame:
    """
    Simulate winding deformation by shifting resonance structure along frequency.

    Parameters
    ----------
    healthy
        Optional baseline frame; if omitted, :func:`generate_healthy_fra` is used.
    peak_shift_factor
        Multiplicative stretch applied to frequency axis before re-sampling.

    Returns
    -------
    pd.DataFrame
        Deformed FRA curve.
    """
    h = healthy if healthy is not None else generate_healthy_fra()
    f = np.asarray(h["Frequency"], dtype=np.float64)
    m = np.asarray(h["Magnitude"], dtype=np.float64)
    xp = f * peak_shift_factor
    m2 = np.interp(f, xp, m) + np.random.default_rng(7).normal(0.0, 0.2, size=f.size)
    return pd.DataFrame({"Frequency": f, "Magnitude": m2})


def generate_insulation_attenuation_fra(
    healthy: pd.DataFrame | None = None,
    *,
    hf_db_loss: float = 5.0,
) -> pd.DataFrame:
    """
    Simulate insulation-related high-frequency attenuation.

    Parameters
    ----------
    healthy
        Optional baseline; default is synthetic healthy data.
    hf_db_loss
        Additional loss (dB) applied with a smooth ramp toward high frequency.

    Returns
    -------
    pd.DataFrame
        Attenuated FRA curve.
    """
    h = healthy if healthy is not None else generate_healthy_fra()
    f = np.asarray(h["Frequency"], dtype=np.float64)
    m = np.asarray(h["Magnitude"], dtype=np.float64)
    fn = (np.log10(f) - np.log10(f.min())) / (np.log10(f.max()) - np.log10(f.min()) + 1e-12)
    ramp = np.clip((fn - 0.55) / 0.45, 0.0, 1.0)
    m2 = m - hf_db_loss * ramp + np.random.default_rng(11).normal(0.0, 0.12, size=f.size)
    return pd.DataFrame({"Frequency": f, "Magnitude": m2})


__all__ = [
    "generate_healthy_fra",
    "generate_insulation_attenuation_fra",
    "generate_winding_deformation_fra",
]
