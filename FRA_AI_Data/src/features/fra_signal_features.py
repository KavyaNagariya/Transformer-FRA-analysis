"""Single-curve FRA feature extraction from frequency/magnitude arrays."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
from scipy.signal import find_peaks

ArrayLike = Union[Sequence[float], np.ndarray]

# Default band edges on log10(frequency): tertiles of the observed range
def _linear_from_db(magnitude_db: np.ndarray) -> np.ndarray:
    """Convert dB magnitude to linear amplitude ratio (positive)."""
    return np.power(10.0, magnitude_db / 20.0)


def _prepare_arrays(
    frequency: ArrayLike,
    magnitude: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float64, drop invalids, sort by frequency."""
    f = np.asarray(frequency, dtype=np.float64).ravel()
    m = np.asarray(magnitude, dtype=np.float64).ravel()
    if f.size != m.size:
        raise ValueError(
            f"frequency and magnitude must have the same length; got {f.size} and {m.size}."
        )
    if f.size < 3:
        raise ValueError("Need at least 3 samples for FRA feature extraction.")

    mask = np.isfinite(f) & np.isfinite(m)
    f, m = f[mask], m[mask]
    if f.size < 3:
        raise ValueError("Too few finite samples after removing NaN/inf.")

    order = np.argsort(f)
    return f[order], m[order]


def _band_masks(frequency: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split frequency axis into low / mid / high by log-frequency tertiles."""
    logf = np.log10(np.clip(frequency, a_min=np.finfo(float).tiny, a_max=None))
    lo, hi = logf.min(), logf.max()
    if hi <= lo:
        # Degenerate band: put everything in mid
        mid = np.ones_like(frequency, dtype=bool)
        return np.zeros_like(mid), mid, np.zeros_like(mid)

    t1 = lo + (hi - lo) / 3.0
    t2 = lo + 2.0 * (hi - lo) / 3.0
    low = logf < t1
    high = logf >= t2
    mid = ~(low | high)
    return low, mid, high


def _band_energy_linear(frequency: np.ndarray, magnitude_db: np.ndarray) -> tuple[float, float, float]:
    """
    Energy per band as sum of squared linear amplitudes (``10^(dB/20)``), per sample.

    This is a discrete proxy for signal energy in each frequency region.
    """
    lin = _linear_from_db(magnitude_db)
    e = lin**2
    low, mid, high = _band_masks(frequency)
    return float(np.sum(e[low])), float(np.sum(e[mid])), float(np.sum(e[high]))


def _curve_smoothness(magnitude: np.ndarray) -> float:
    """
    Smoothness score: higher means a smoother curve (smaller second differences).

    Uses inverse of mean absolute second derivative, mapped to a stable scale.
    """
    if magnitude.size < 3:
        return 0.0
    d2 = np.diff(magnitude, n=2)
    rough = float(np.mean(np.abs(d2)))
    return float(1.0 / (1.0 + rough))


def _interpolate_to_base(
    f_ref: np.ndarray,
    m_ref: np.ndarray,
    f_query: np.ndarray,
) -> np.ndarray:
    """Linear interpolate reference magnitude onto query frequencies; NaN outside overlap."""
    if f_ref.size < 2:
        raise ValueError("Reference signal needs at least two points.")
    f_r, m_r = _prepare_arrays(f_ref, m_ref)
    out = np.interp(f_query, f_r, m_r)
    out[(f_query < f_r.min()) | (f_query > f_r.max())] = np.nan
    return out.astype(np.float64)


def extract_fra_signal_features(
    frequency: ArrayLike,
    magnitude: ArrayLike,
    *,
    reference_frequency: Optional[ArrayLike] = None,
    reference_magnitude: Optional[ArrayLike] = None,
    peak_prominence: Optional[float] = None,
    peak_distance: Optional[int] = None,
) -> dict[str, Any]:
    """
    Extract descriptive features from a single FRA magnitude trace.

    Peaks are detected with :func:`scipy.signal.find_peaks` on the magnitude series.
    Frequency bands (low / mid / high) are log-frequency tertiles of the observed range.

    Parameters
    ----------
    frequency
        Frequency samples (Hz), any length ≥ 3.
    magnitude
        Magnitude samples (typically dB), same length as ``frequency``.
    reference_frequency, reference_magnitude
        Optional reference curve for Pearson correlation; interpolated to this curve's
        frequency grid where ranges overlap. Both must be provided together.
    peak_prominence
        Passed to ``find_peaks``; default scales with magnitude standard deviation.
    peak_distance
        Minimum index distance between peaks; default scales with array length.

    Returns
    -------
    dict
        Keys include:
        ``n_peaks``, ``peak_frequencies``, ``peak_amplitudes``, ``mean_magnitude``,
        ``std_magnitude``, ``energy_low_band``, ``energy_mid_band``, ``energy_high_band``,
        ``curve_smoothness``, and ``correlation_reference`` (``None`` if no reference).
    """
    f, mag = _prepare_arrays(frequency, magnitude)

    if reference_frequency is not None and reference_magnitude is not None:
        f_ref = np.asarray(reference_frequency, dtype=np.float64).ravel()
        m_ref = np.asarray(reference_magnitude, dtype=np.float64).ravel()
    elif reference_frequency is None and reference_magnitude is None:
        f_ref = m_ref = None
    else:
        raise ValueError(
            "Provide both reference_frequency and reference_magnitude, or neither."
        )

    n = mag.size
    std_m = float(np.std(mag))
    if peak_prominence is None:
        peak_prominence = max(0.1 * std_m, 1e-9) if std_m > 0 else 1e-9
    if peak_distance is None:
        peak_distance = max(1, n // 100)

    peaks, props = find_peaks(
        mag,
        prominence=peak_prominence,
        distance=peak_distance,
    )

    n_peaks = int(peaks.size)
    peak_frequencies = f[peaks].astype(np.float64)
    peak_amplitudes = mag[peaks].astype(np.float64)

    mean_magnitude = float(np.mean(mag))
    std_magnitude = float(np.std(mag))

    e_low, e_mid, e_high = _band_energy_linear(f, mag)
    smooth = _curve_smoothness(mag)

    corr_ref: Optional[float]
    if f_ref is not None:
        m_on_grid = _interpolate_to_base(f_ref, m_ref, f)
        valid = np.isfinite(m_on_grid)
        if np.count_nonzero(valid) >= 3:
            a, b = mag[valid], m_on_grid[valid]
            if np.std(a) > 0 and np.std(b) > 0:
                c = np.corrcoef(a, b)[0, 1]
                corr_ref = float(c) if np.isfinite(c) else None
            else:
                corr_ref = None
        else:
            corr_ref = None
    else:
        corr_ref = None

    return {
        "n_peaks": n_peaks,
        "peak_frequencies": peak_frequencies,
        "peak_amplitudes": peak_amplitudes,
        "mean_magnitude": mean_magnitude,
        "std_magnitude": std_magnitude,
        "energy_low_band": e_low,
        "energy_mid_band": e_mid,
        "energy_high_band": e_high,
        "curve_smoothness": smooth,
        "correlation_reference": corr_ref,
    }


def fra_features_to_vector(
    features: Mapping[str, Any],
    *,
    include_peaks: bool = True,
    max_peaks: int = 32,
) -> np.ndarray:
    """
    Flatten scalar features into a 1-D ``float64`` vector for ML pipelines.

    Peak frequencies and amplitudes are truncated/padded with NaN to ``max_peaks``.

    Parameters
    ----------
    features
        Output of :func:`extract_fra_signal_features`.
    include_peaks
        If False, only scalar summary stats are included (no per-peak arrays).
    max_peaks
        Fixed length for peak frequency and amplitude slots.

    Returns
    -------
    np.ndarray
        1-D vector; peak blocks use NaN padding when ``n_peaks < max_peaks``.
    """
    parts: list[np.ndarray] = []

    parts.append(
        np.array(
            [
                float(features["n_peaks"]),
                features["mean_magnitude"],
                features["std_magnitude"],
                features["energy_low_band"],
                features["energy_mid_band"],
                features["energy_high_band"],
                features["curve_smoothness"],
            ],
            dtype=np.float64,
        )
    )

    c = features.get("correlation_reference")
    parts.append(np.array([np.nan if c is None else float(c)], dtype=np.float64))

    if include_peaks:
        pf = np.asarray(features["peak_frequencies"], dtype=np.float64).ravel()
        pa = np.asarray(features["peak_amplitudes"], dtype=np.float64).ravel()
        pf = pf[:max_peaks]
        pa = pa[:max_peaks]
        if pf.size < max_peaks:
            pad = max_peaks - pf.size
            pf = np.pad(pf, (0, pad), constant_values=np.nan)
            pa = np.pad(pa, (0, pad), constant_values=np.nan)
        parts.append(pf)
        parts.append(pa)

    return np.concatenate(parts)


def feature_dict_for_ui(
    frequency: ArrayLike,
    magnitude: ArrayLike,
    *,
    reference_frequency: Optional[ArrayLike] = None,
    reference_magnitude: Optional[ArrayLike] = None,
) -> dict[str, Any]:
    """
    Build a UI- and report-friendly feature dictionary with stable key names.

    Parameters
    ----------
    frequency, magnitude
        FRA samples (Hz, dB or linear per your data convention).
    reference_frequency, reference_magnitude
        Optional reference curve for correlation-based context.

    Returns
    -------
    dict
        Keys: ``num_peaks``, ``peak_frequencies`` (list of float), ``mean_magnitude``,
        ``std_dev``, ``low_freq_energy``, ``mid_freq_energy``, ``high_freq_energy``.
    """
    raw = extract_fra_signal_features(
        frequency,
        magnitude,
        reference_frequency=reference_frequency,
        reference_magnitude=reference_magnitude,
    )
    pf = np.asarray(raw["peak_frequencies"], dtype=np.float64).ravel()
    return {
        "num_peaks": int(raw["n_peaks"]),
        "peak_frequencies": [float(x) for x in pf.tolist()],
        "mean_magnitude": float(raw["mean_magnitude"]),
        "std_dev": float(raw["std_magnitude"]),
        "low_freq_energy": float(raw["energy_low_band"]),
        "mid_freq_energy": float(raw["energy_mid_band"]),
        "high_freq_energy": float(raw["energy_high_band"]),
    }


def generate_feature_insights(
    test_summary: Mapping[str, Any],
    *,
    reference_summary: Optional[Mapping[str, Any]] = None,
    std_threshold_ratio: float = 1.5,
    peak_shift_hz_threshold: float = 0.02,
) -> list[str]:
    """
    Produce short human-readable insight strings from scalar FRA summaries.

    Parameters
    ----------
    test_summary
        Output of :func:`feature_dict_for_ui` for the test curve.
    reference_summary
        Optional reference summary for comparative phrases.
    std_threshold_ratio
        Flag ``High variation`` when test ``std_dev`` exceeds this times the reference
        ``std_dev`` (if reference is given), else when ``std_dev`` is large vs mean scale.
    peak_shift_hz_threshold
        Minimum relative peak-frequency shift to emit ``Peak shift observed``.

    Returns
    -------
    list[str]
        Non-empty list of short diagnostic phrases for dashboards.
    """
    insights: list[str] = []
    std_t = float(test_summary.get("std_dev", 0.0))
    mean_t = abs(float(test_summary.get("mean_magnitude", 0.0))) + 1e-9

    if reference_summary is not None:
        std_r = float(reference_summary.get("std_dev", 0.0)) + 1e-12
        if std_t > std_threshold_ratio * std_r:
            insights.append("High variation detected")
        peaks_t = np.asarray(test_summary.get("peak_frequencies", []), dtype=np.float64).ravel()
        peaks_r = np.asarray(reference_summary.get("peak_frequencies", []), dtype=np.float64).ravel()
        if peaks_t.size and peaks_r.size:
            # Compare dominant peaks (first by frequency order in extract)
            dt = float(peaks_t[0])
            dr = float(peaks_r[0])
            denom = max(abs(dr), 1.0)
            if abs(dt - dr) / denom > peak_shift_hz_threshold:
                insights.append("Peak shift observed")
    else:
        if std_t / mean_t > 0.15:
            insights.append("High variation detected")

    if not insights:
        insights.append("Signature appears stable relative to extracted statistics")

    return insights
