"""
CardioVision — Rule-Based Signal Auditor
Automated identification of signal-level quality problems.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AuditReport:
    """Structured audit report for an ECG signal."""
    record_id: str = ""
    total_samples: int = 0
    duration_sec: float = 0.0
    sampling_rate: float = 0.0

    # Findings
    missing_count: int = 0
    missing_pct: float = 0.0
    duplicate_count: int = 0
    duplicate_pct: float = 0.0
    flat_segments: int = 0
    amplitude_outliers: int = 0
    sampling_rate_ok: bool = True

    issues: List[str] = field(default_factory=list)
    severity: str = "PASS"  # PASS, WARNING, CRITICAL

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "total_samples": self.total_samples,
            "duration_sec": round(self.duration_sec, 2),
            "sampling_rate": self.sampling_rate,
            "missing_count": self.missing_count,
            "missing_pct": round(self.missing_pct, 2),
            "duplicate_count": self.duplicate_count,
            "duplicate_pct": round(self.duplicate_pct, 2),
            "flat_segments": self.flat_segments,
            "amplitude_outliers": self.amplitude_outliers,
            "sampling_rate_ok": self.sampling_rate_ok,
            "issues": self.issues,
            "severity": self.severity,
        }


class SignalAuditor:
    """
    Rule-based quality auditor for ECG signals.
    Checks for missing data, duplicates, flat-line segments,
    amplitude anomalies, and sampling-rate consistency.
    """

    def __init__(
        self,
        expected_fs: float = 360.0,
        fs_tolerance: float = 0.05,
        flat_threshold: int = 50,
        amplitude_range: tuple = (-5.0, 5.0),
        outlier_zscore: float = 5.0,
    ):
        self.expected_fs = expected_fs
        self.fs_tolerance = fs_tolerance
        self.flat_threshold = flat_threshold
        self.amplitude_range = amplitude_range
        self.outlier_zscore = outlier_zscore

    def check_missing(self, signal: np.ndarray) -> tuple:
        """Check for NaN and Inf values."""
        mask = np.isnan(signal) | np.isinf(signal)
        count = int(np.sum(mask))
        pct = 100.0 * count / len(signal) if len(signal) > 0 else 0.0
        return count, pct

    def check_duplicates(self, signal: np.ndarray) -> tuple:
        """
        Check for consecutive duplicate values (potential sensor freeze).
        A run of identical values longer than flat_threshold is suspicious.
        """
        if len(signal) < 2:
            return 0, 0.0

        diffs = np.diff(signal)
        zero_diffs = diffs == 0
        count = int(np.sum(zero_diffs))
        pct = 100.0 * count / len(signal)
        return count, pct

    def check_flat_segments(self, signal: np.ndarray) -> int:
        """
        Count the number of flat-line segments (runs of identical values
        longer than flat_threshold).
        """
        if len(signal) < self.flat_threshold:
            return 0

        flat_count = 0
        run_length = 1
        for i in range(1, len(signal)):
            if signal[i] == signal[i - 1]:
                run_length += 1
            else:
                if run_length >= self.flat_threshold:
                    flat_count += 1
                run_length = 1
        if run_length >= self.flat_threshold:
            flat_count += 1
        return flat_count

    def check_amplitude(self, signal: np.ndarray) -> int:
        """Count samples outside expected physiological amplitude range."""
        clean = signal[~np.isnan(signal) & ~np.isinf(signal)]
        if len(clean) == 0:
            return 0
        lo, hi = self.amplitude_range
        outliers = int(np.sum((clean < lo) | (clean > hi)))
        return outliers

    def check_sampling_rate(self, actual_fs: float) -> bool:
        """Verify sampling rate is within tolerance of expected."""
        if self.expected_fs == 0:
            return True
        ratio = abs(actual_fs - self.expected_fs) / self.expected_fs
        return ratio <= self.fs_tolerance

    def audit(
        self,
        signal: np.ndarray,
        record_id: str = "unknown",
        actual_fs: Optional[float] = None,
    ) -> AuditReport:
        """
        Run all quality checks and produce a structured AuditReport.

        Parameters
        ----------
        signal : 1-D ECG signal array.
        record_id : Identifier for the record being audited.
        actual_fs : Actual sampling rate (if known).

        Returns
        -------
        AuditReport with all findings and severity level.
        """
        fs = actual_fs or self.expected_fs
        report = AuditReport(
            record_id=record_id,
            total_samples=len(signal),
            duration_sec=len(signal) / fs if fs > 0 else 0.0,
            sampling_rate=fs,
        )

        # Missing data
        report.missing_count, report.missing_pct = self.check_missing(signal)
        if report.missing_pct > 0:
            report.issues.append(
                f"Missing data: {report.missing_count} samples ({report.missing_pct:.1f}%)"
            )

        # Duplicates
        report.duplicate_count, report.duplicate_pct = self.check_duplicates(signal)
        if report.duplicate_pct > 20:
            report.issues.append(
                f"High duplicate rate: {report.duplicate_pct:.1f}% consecutive identical values"
            )

        # Flat segments
        report.flat_segments = self.check_flat_segments(signal)
        if report.flat_segments > 0:
            report.issues.append(
                f"Flat-line segments detected: {report.flat_segments} segments ≥{self.flat_threshold} samples"
            )

        # Amplitude
        report.amplitude_outliers = self.check_amplitude(signal)
        if report.amplitude_outliers > 0:
            report.issues.append(
                f"Amplitude outliers: {report.amplitude_outliers} samples outside {self.amplitude_range}"
            )

        # Sampling rate
        if actual_fs is not None:
            report.sampling_rate_ok = self.check_sampling_rate(actual_fs)
            if not report.sampling_rate_ok:
                report.issues.append(
                    f"Sampling rate mismatch: expected {self.expected_fs} Hz, got {actual_fs} Hz"
                )

        # Severity
        if report.missing_pct > 20 or not report.sampling_rate_ok or report.flat_segments > 3:
            report.severity = "CRITICAL"
        elif len(report.issues) > 0:
            report.severity = "WARNING"
        else:
            report.severity = "PASS"

        return report
