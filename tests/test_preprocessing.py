"""
CardioVision — Preprocessing Unit Tests
Tests for filters, imputation, and auditing modules.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.filters import butterworth_bandpass, notch_filter, apply_all_filters
from src.preprocessing.imputer import detect_missing, handle_missing_values
from src.preprocessing.auditor import SignalAuditor, AuditReport


class TestFilters:
    """Tests for the signal filtering module."""

    def setup_method(self):
        """Create test signals."""
        self.fs = 360
        t = np.arange(0, 2, 1/self.fs)  # 2 seconds
        # Clean ECG-like signal (10 Hz component)
        self.clean = np.sin(2 * np.pi * 10 * t)
        # Add 60 Hz power-line noise
        self.noisy = self.clean + 0.5 * np.sin(2 * np.pi * 60 * t)
        # Add baseline wander (0.2 Hz)
        self.wandering = self.clean + 2.0 * np.sin(2 * np.pi * 0.2 * t)

    def test_butterworth_preserves_shape(self):
        """Butterworth filter should not change signal length."""
        filtered = butterworth_bandpass(self.noisy, fs=self.fs)
        assert len(filtered) == len(self.noisy)

    def test_butterworth_removes_baseline_wander(self):
        """Bandpass filter should reduce low-frequency baseline drift."""
        filtered = butterworth_bandpass(self.wandering, fs=self.fs)
        # The DC / very low frequency component should be attenuated
        fft_orig = np.abs(np.fft.rfft(self.wandering))
        fft_filt = np.abs(np.fft.rfft(filtered))
        # Energy below 1 Hz should decrease
        low_bins = int(1.0 * len(self.wandering) / self.fs)
        assert np.sum(fft_filt[:low_bins]) < np.sum(fft_orig[:low_bins])

    def test_notch_removes_powerline(self):
        """Notch filter should attenuate 60 Hz interference."""
        filtered = notch_filter(self.noisy, freq=60, fs=self.fs)
        fft_orig = np.abs(np.fft.rfft(self.noisy))
        fft_filt = np.abs(np.fft.rfft(filtered))
        # Find the 60 Hz bin
        bin_60 = int(60 * len(self.noisy) / self.fs)
        assert fft_filt[bin_60] < fft_orig[bin_60] * 0.5

    def test_apply_all_filters(self):
        """Combined filter pipeline should work end to end."""
        combined = self.wandering + 0.5 * np.sin(2 * np.pi * 60 * np.arange(len(self.wandering)) / self.fs)
        filtered = apply_all_filters(combined, fs=self.fs)
        assert len(filtered) == len(combined)
        assert not np.any(np.isnan(filtered))

    def test_short_signal_handling(self):
        """Filters should handle very short signals gracefully."""
        short = np.array([1.0, 2.0, 3.0])
        result = butterworth_bandpass(short, fs=self.fs)
        assert len(result) == 3

    def test_empty_signal(self):
        """Should handle empty arrays."""
        empty = np.array([])
        result = apply_all_filters(empty, fs=self.fs)
        assert len(result) == 0


class TestImputer:
    """Tests for the missing value imputation module."""

    def test_detect_no_missing(self):
        """Signal without NaN should report 0% missing."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask, pct = detect_missing(signal)
        assert pct == 0.0
        assert not np.any(mask)

    def test_detect_with_nan(self):
        """Should correctly detect NaN values."""
        signal = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        mask, pct = detect_missing(signal)
        assert pct == 40.0
        assert np.sum(mask) == 2

    def test_detect_with_inf(self):
        """Should detect Inf values as missing."""
        signal = np.array([1.0, np.inf, 3.0, -np.inf, 5.0])
        mask, pct = detect_missing(signal)
        assert np.sum(mask) == 2

    def test_linear_imputation(self):
        """Linear interpolation should fill NaN gaps."""
        signal = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result, info = handle_missing_values(signal, method="linear")
        assert not np.any(np.isnan(result))
        assert abs(result[1] - 2.0) < 0.01
        assert abs(result[3] - 4.0) < 0.01

    def test_spline_imputation(self):
        """Spline interpolation should fill NaN gaps smoothly."""
        signal = np.array([0, 1, np.nan, 3, 4, np.nan, 6, 7, 8, 9.0])
        result, info = handle_missing_values(signal, method="spline")
        assert not np.any(np.isnan(result))
        assert info["method"] == "spline"

    def test_reliability_flag(self):
        """Signal with >30% missing should be flagged unreliable."""
        signal = np.array([1.0, np.nan, np.nan, np.nan, 5.0])  # 60% missing
        _, info = handle_missing_values(signal, max_gap_pct=30.0)
        assert not info["reliable"]

    def test_no_imputation_needed(self):
        """Clean signal should pass through unchanged."""
        signal = np.array([1.0, 2.0, 3.0])
        result, info = handle_missing_values(signal)
        np.testing.assert_array_equal(result, signal)
        assert info["action"] == "none_needed"


class TestAuditor:
    """Tests for the rule-based signal auditor."""

    def setup_method(self):
        self.auditor = SignalAuditor(expected_fs=360.0)

    def test_clean_signal_passes(self):
        """A clean signal should get PASS severity."""
        signal = np.sin(np.linspace(0, 10, 3600))
        report = self.auditor.audit(signal, record_id="test_clean")
        assert report.severity == "PASS"
        assert len(report.issues) == 0

    def test_missing_data_detection(self):
        """Should detect NaN values."""
        signal = np.ones(1000)
        signal[100] = np.nan
        signal[500] = np.nan
        report = self.auditor.audit(signal, record_id="test_missing")
        assert report.missing_count == 2

    def test_flat_segment_detection(self):
        """Should detect flat-line segments."""
        signal = np.random.randn(1000)
        signal[200:310] = 0.0  # 110-sample flat segment
        report = self.auditor.audit(signal, record_id="test_flat")
        assert report.flat_segments >= 1

    def test_sampling_rate_mismatch(self):
        """Should flag mismatched sampling rate."""
        signal = np.random.randn(1000)
        report = self.auditor.audit(signal, record_id="test_fs", actual_fs=250.0)
        assert not report.sampling_rate_ok

    def test_critical_severity(self):
        """High missing rate should trigger CRITICAL."""
        signal = np.ones(100)
        signal[:25] = np.nan  # 25% missing
        report = self.auditor.audit(signal, record_id="test_crit")
        assert report.severity in ("WARNING", "CRITICAL")

    def test_report_to_dict(self):
        """AuditReport should serialize to dict."""
        signal = np.random.randn(360)
        report = self.auditor.audit(signal, record_id="test_dict")
        d = report.to_dict()
        assert "record_id" in d
        assert "severity" in d
        assert d["record_id"] == "test_dict"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
