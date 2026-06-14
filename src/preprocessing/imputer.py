"""
CardioVision — Missing Value Imputation
Handles NaN/Inf detection and interpolation for ECG signals.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple


def detect_missing(signal: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Detect missing (NaN) and infinite values in an ECG signal.

    Returns
    -------
    mask : Boolean array — True where values are missing/invalid.
    pct  : Percentage of missing values.
    """
    mask = np.isnan(signal) | np.isinf(signal)
    pct = 100.0 * np.sum(mask) / len(signal) if len(signal) > 0 else 0.0
    return mask, pct


def interpolate_linear(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fill missing values using linear interpolation.
    Edge cases (leading/trailing NaNs) are forward/backward filled.
    """
    result = signal.copy()
    if not np.any(mask):
        return result

    valid_idx = np.where(~mask)[0]
    missing_idx = np.where(mask)[0]

    if len(valid_idx) < 2:
        # Not enough valid points — fill with mean or zero
        fill_val = np.nanmean(signal) if np.any(~mask) else 0.0
        result[mask] = fill_val
        return result

    result[missing_idx] = np.interp(missing_idx, valid_idx, result[valid_idx])
    return result


def interpolate_spline(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fill missing values using cubic spline interpolation.
    Falls back to linear if too few valid points.
    """
    result = signal.copy()
    if not np.any(mask):
        return result

    valid_idx = np.where(~mask)[0]
    missing_idx = np.where(mask)[0]

    if len(valid_idx) < 4:
        return interpolate_linear(signal, mask)

    cs = CubicSpline(valid_idx, result[valid_idx], extrapolate=True)
    result[missing_idx] = cs(missing_idx)
    return result


def handle_missing_values(
    signal: np.ndarray,
    method: str = "linear",
    max_gap_pct: float = 30.0,
) -> Tuple[np.ndarray, dict]:
    """
    Full missing-value handling pipeline.

    Parameters
    ----------
    signal : Raw 1-D ECG signal (may contain NaN/Inf).
    method : Interpolation method — 'linear' or 'spline'.
    max_gap_pct : If missing % exceeds this, flag as unreliable.

    Returns
    -------
    imputed : Cleaned signal with missing values filled.
    info : Dict with imputation statistics.
    """
    mask, pct_missing = detect_missing(signal)

    info = {
        "total_samples": len(signal),
        "missing_count": int(np.sum(mask)),
        "missing_pct": round(pct_missing, 2),
        "method": method,
        "reliable": pct_missing <= max_gap_pct,
    }

    if not np.any(mask):
        info["action"] = "none_needed"
        return signal.copy(), info

    if pct_missing > max_gap_pct:
        info["action"] = "imputed_but_flagged_unreliable"

    if method == "spline":
        imputed = interpolate_spline(signal, mask)
    else:
        imputed = interpolate_linear(signal, mask)

    info["action"] = info.get("action", "imputed")
    return imputed, info
