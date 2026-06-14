"""
CardioVision — Digital Signal Filters
Butterworth bandpass and Notch filters for ECG denoising.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    BUTTER_LOWCUT, BUTTER_HIGHCUT, BUTTER_ORDER,
    NOTCH_FREQ, NOTCH_QUALITY, SAMPLING_RATE,
)


def butterworth_bandpass(
    signal: np.ndarray,
    lowcut: float = BUTTER_LOWCUT,
    highcut: float = BUTTER_HIGHCUT,
    fs: float = SAMPLING_RATE,
    order: int = BUTTER_ORDER,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to remove baseline wander
    (low-frequency drift) and high-frequency muscle noise.

    Parameters
    ----------
    signal : 1-D array of ECG samples.
    lowcut : Lower cutoff frequency in Hz (removes baseline wander below this).
    highcut : Upper cutoff frequency in Hz (removes HF noise above this).
    fs : Sampling rate in Hz.
    order : Filter order (higher = sharper roll-off, but risk of ringing).

    Returns
    -------
    Filtered signal with the same length as input.
    """
    if len(signal) < 3 * order:
        return signal.copy()

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Clamp to valid range
    low = max(low, 1e-5)
    high = min(high, 1.0 - 1e-5)

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal, padlen=min(3 * max(len(b), len(a)), len(signal) - 1))


def notch_filter(
    signal: np.ndarray,
    freq: float = NOTCH_FREQ,
    fs: float = SAMPLING_RATE,
    quality: float = NOTCH_QUALITY,
) -> np.ndarray:
    """
    Apply a notch (band-stop) filter to remove power-line interference
    at the specified frequency (typically 50 Hz or 60 Hz).

    Parameters
    ----------
    signal : 1-D array of ECG samples.
    freq : Frequency to notch out (Hz).
    fs : Sampling rate (Hz).
    quality : Quality factor — higher means narrower notch.

    Returns
    -------
    Filtered signal.
    """
    if len(signal) < 6:
        return signal.copy()

    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal, padlen=min(3 * max(len(b), len(a)), len(signal) - 1))


def apply_all_filters(
    signal: np.ndarray,
    fs: float = SAMPLING_RATE,
    apply_bandpass: bool = True,
    apply_notch: bool = True,
    notch_freqs: Optional[list] = None,
) -> np.ndarray:
    """
    Apply the full denoising pipeline: bandpass then notch filter(s).

    Parameters
    ----------
    signal : Raw 1-D ECG signal.
    fs : Sampling rate (Hz).
    apply_bandpass : Whether to apply Butterworth bandpass.
    apply_notch : Whether to apply notch filter(s).
    notch_freqs : List of frequencies to notch. Defaults to [NOTCH_FREQ].

    Returns
    -------
    Fully denoised signal.
    """
    filtered = signal.copy().astype(np.float64)

    if apply_bandpass:
        filtered = butterworth_bandpass(filtered, fs=fs)

    if apply_notch:
        freqs = notch_freqs or [NOTCH_FREQ]
        for f in freqs:
            if f < fs / 2:  # Only notch below Nyquist
                filtered = notch_filter(filtered, freq=f, fs=fs)

    return filtered
